"""Scene-text supervision with image-only deployment and explicit paired mode.

The caption condition comes from a frozen CLIP text encoder, never the task
question or gold completion. An image-side predictor learns that representation
for deployments without captions. Paired inference is a different protocol.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

METHOD = 'scene_text_distilled_moralclip_encoder_v2'
INDEPENDENT = 'smid-independent-image-text-v2'
PAIRED = 'smid-paired-caption-assisted-v2'


@dataclass(frozen=True)
class SceneConfig:
    vision_dim: int = 768
    text_dim: int = 512
    rank: int = 32
    residual_scale: float = .1
    feature_layer: int = -2
    condition_source: str = 'frozen_moralclip_scene_text_or_image_prediction'

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.vision_dim, self.text_dim, self.rank)):
            raise ValueError('Scene guidance dimensions must be positive integers')
        if not 0 < self.residual_scale <= 1:
            raise ValueError('Invalid scene residual scale')
        if self.condition_source != 'frozen_moralclip_scene_text_or_image_prediction':
            raise ValueError('Invalid scene condition source')


@dataclass(frozen=True)
class SceneContext:
    mode: str = 'predicted'
    text_features: torch.Tensor | None = None

    def __post_init__(self):
        if self.mode not in ('predicted', 'paired'):
            raise ValueError('Scene mode must be predicted or paired')
        if self.mode == 'paired' and self.text_features is None:
            raise ValueError('Paired mode requires actual scene text features')


class SceneResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        from .text_guided_vision import TextGuidedResidual
        self.config = config
        self.predictor = nn.Sequential(nn.LayerNorm(config.vision_dim),
            nn.Linear(config.vision_dim, config.rank), nn.GELU(),
            nn.Linear(config.rank, config.text_dim))
        self.residual = TextGuidedResidual(config)

    def forward(self, spatial, context):
        if not isinstance(context, SceneContext):
            raise ValueError('Explicit scene context required')
        features = F.normalize(self.predictor(spatial.float().mean(1)), dim=-1)
        target = context.text_features
        if target is not None:
            target = target.detach().to(device=features.device, dtype=features.dtype)
            if target.shape != features.shape or not torch.isfinite(target).all():
                raise ValueError('Scene text feature shape/value mismatch')
            if (target.norm(dim=-1) < 1e-8).any():
                raise ValueError('Empty scene text representation')
            target = F.normalize(target, dim=-1)
        # In predicted mode target is a loss target only, never an input to FiLM.
        condition = features if context.mode == 'predicted' else target
        adapted = self.residual(spatial, condition)
        alignment = (1 - (features * target).sum(-1)).mean() if target is not None else None
        return adapted, alignment


class SceneEncoderBlock(nn.Module):
    def __init__(self, base_block, adapter):
        super().__init__()
        self.base_block, self.adapter = base_block, adapter
        self._active, self._context, self._enabled = False, None, True
        self.last_penalty = self.last_alignment = None

    @contextmanager
    def conditioned(self, context=None, *, enabled=True):
        if self._active:
            raise RuntimeError('Nested scene contexts are unsupported')
        if enabled and not isinstance(context, SceneContext):
            raise ValueError('Scene guidance requires a scene context')
        self._active, self._context, self._enabled = True, context, enabled
        self.last_penalty = self.last_alignment = None
        try:
            yield
        finally:
            self._active, self._context = False, None
            self.last_penalty = self.last_alignment = None

    def forward(self, *args, **kwargs):
        if not self._active:
            raise RuntimeError('Vision forward requires an explicit scene context')
        output = self.base_block(*args, **kwargs)
        if not self._enabled:
            return output
        spatial = output[0][:, 1:]
        adapted, self.last_alignment = self.adapter(spatial, self._context)
        self.last_penalty = (adapted-spatial).float().square().mean() / (
            spatial.detach().float().square().mean()+1e-6)
        return (torch.cat((output[0][:, :1], adapted), 1), *output[1:])


def install_scene_guidance(model, rank=32, *, config=None):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    config = config or SceneConfig(vision_dim=base.config.vision_config.hidden_size, rank=rank)
    layers = base.vision_tower.vision_model.encoder.layers
    if (base.config.vision_config.hidden_size != config.vision_dim or
            base.config.vision_feature_layer != config.feature_layer or
            base.config.vision_feature_select_strategy != 'default'):
        raise ValueError('Scene adapter incompatible with visual feature selection')
    if not -(len(layers)+1) <= config.feature_layer <= len(layers):
        raise ValueError('Visual feature layer out of range')
    index = list(range(len(layers)+1))[config.feature_layer]-1
    if index < 0 or isinstance(layers[index], SceneEncoderBlock) or hasattr(layers[index], 'adapter'):
        raise ValueError('Invalid or already wrapped scene layer')
    reference = next(layers[index].parameters())
    model.requires_grad_(False)
    block = SceneEncoderBlock(layers[index], SceneResidual(config).to(device=reference.device))
    layers[index] = block
    return block, index


def caption_index(raw, *, split):
    """Accept only actual caption0 text rows; never read completions as captions."""
    result = {}
    ids = set()
    pairs = {}
    for row in raw:
        if row.get('split') != split or row['id'] in ids:
            raise ValueError('Cross-split or duplicate scene record')
        ids.add(row['id'])
        if row['modality'] not in ('image', 'text'):
            raise ValueError('Unexpected scene modality')
        key = (row['scene_id'], row['task'])
        if row['modality'] in pairs.setdefault(key, set()):
            raise ValueError('Duplicate scene/task/modality')
        pairs[key].add(row['modality'])
        if row['modality'] == 'text':
            caption = row.get('caption')
            if row.get('caption_index') != 0 or not isinstance(caption, str) or not caption.strip():
                raise ValueError('A nonempty caption0 is required')
            if row['scene_id'] in result and result[row['scene_id']] != caption:
                raise ValueError('Scene captions differ across tasks')
            result[row['scene_id']] = caption
    if not pairs or any(value != {'image', 'text'} for value in pairs.values()):
        raise ValueError('Unpaired scene records')
    return result


def paired_text_rows(image_rows, normalized_rows):
    """Pair by scene AND task; supervision is used for training only."""
    index = {r['id']: r for r in normalized_rows}
    result = []
    for image in image_rows:
        key = f'{image["scene_id"]}::{image["task"]}::text'
        text = index.get(key)
        if (image['modality'] != 'image' or text is None or text['modality'] != 'text' or
                text['split'] != image['split']):
            raise ValueError('Missing or cross-split text teacher')
        if image['target_json'] is None or text['target_json'] != image['target_json']:
            raise ValueError('Training pair targets differ or are absent')
        result.append(text)
    return result


class SceneTextEncoder:
    """Frozen MoralCLIP text projection; no question, task ID, or gold answer."""
    def __init__(self, directory, *, device='cuda'):
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model, info = CLIPTextModelWithProjection.from_pretrained(
            directory, local_files_only=True, output_loading_info=True)
        if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError('Scene text encoder has missing or incompatible pretrained weights')
        self.model.to(device)
        self.model.requires_grad_(False).eval()
        self.device = device
        self.cache = {}
        self.truncated_captions = set()

    @torch.no_grad()
    def encode(self, captions):
        if not captions or any(not isinstance(t, str) or not t.strip() for t in captions):
            raise ValueError('Scene descriptions must be nonempty strings')
        missing = list(dict.fromkeys(t for t in captions if t not in self.cache))
        if missing:
            limit = self.model.config.max_position_embeddings
            lengths = self.tokenizer(missing, truncation=False)['input_ids']
            self.truncated_captions.update(t for t, ids in zip(missing, lengths) if len(ids) > limit)
            tokens = self.tokenizer(missing, padding=True, truncation=True,
                                    max_length=limit, return_tensors='pt').to(self.device)
            features = F.normalize(self.model(**tokens).text_embeds.float(), dim=-1).cpu()
            self.cache.update(zip(missing, features))
        return torch.stack([self.cache[t] for t in captions]).to(self.device)


def answer_logits(logits, labels):
    """Align image/text predictions by supervised answer positions, never padding."""
    if logits.shape[:2] != labels.shape or labels.ndim != 2:
        raise ValueError('Logits/labels shape mismatch')
    result = []
    for logit, target in zip(logits[:, :-1], labels[:, 1:]):
        mask = target != -100
        if not mask.any():
            raise ValueError('Empty shifted supervision')
        result.append((logit[mask].float(), target[mask]))
    return result


def supervised_and_distilled(student_logits, labels, teacher, *, temperature=2.):
    if not temperature > 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError('Temperature must be positive and finite')
    student = answer_logits(student_logits, labels)
    if len(student) != len(teacher):
        raise ValueError('Teacher/student batch mismatch')
    ce, kl, gates = [], [], []
    for (s, target), (t, teacher_target) in zip(student, teacher):
        if t.shape != s.shape or not torch.equal(target, teacher_target.to(target.device)):
            raise ValueError('Teacher/student answer tokens differ')
        t = t.detach().to(s.device)
        image_ce = F.cross_entropy(s, target)
        text_ce = F.cross_entropy(t, target)
        # Reject a teacher with worse supervised NLL than the current image model.
        # This is a training-only gate, not a claim of free-generation correctness.
        gate = (text_ce <= image_ce.detach()).float()
        distill = F.kl_div(F.log_softmax(s/temperature, -1),
            F.softmax(t/temperature, -1), reduction='batchmean') * temperature**2
        ce.append(image_ce)
        kl.append(gate*distill)
        gates.append(gate)
    return torch.stack(ce).mean(), torch.stack(kl).mean(), torch.stack(gates).mean()


def save_scene_guidance(path, block, metadata):
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in block.adapter.state_dict().items()},
              path/'guidance.safetensors')
    payload = {'method': METHOD, 'config': asdict(block.adapter.config),
        'weights_sha256': hashlib.sha256((path/'guidance.safetensors').read_bytes()).hexdigest(),
        'metadata': metadata}
    (path/'guidance.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    return payload


def load_scene_guidance(path, block):
    from safetensors.torch import load_file
    path = Path(path)
    payload = json.loads((path/'guidance.json').read_text())
    if payload['method'] != METHOD or payload['config'] != asdict(block.adapter.config):
        raise ValueError('Scene guidance architecture mismatch')
    if hashlib.sha256((path/'guidance.safetensors').read_bytes()).hexdigest() != payload['weights_sha256']:
        raise ValueError('Scene guidance SHA-256 mismatch')
    block.adapter.load_state_dict(load_file(path/'guidance.safetensors'), strict=True)
    return payload
