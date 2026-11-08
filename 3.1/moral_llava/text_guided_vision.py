"""Text-conditioned residuals *inside* the selected MoralCLIP encoder block.

Conditioning uses the human question only, never the completion or paired
caption. Frozen language input embeddings provide the text representation.
This v1 represents task semantics, not image-generated scene descriptions.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

METHOD = 'question_conditioned_moralclip_encoder_v1'


@dataclass(frozen=True)
class GuidanceConfig:
    vision_dim: int = 768
    text_dim: int = 4096
    rank: int = 32
    residual_scale: float = 0.1
    feature_layer: int = -2
    condition_source: str = 'human_question_frozen_input_embeddings'

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.vision_dim, self.text_dim, self.rank)):
            raise ValueError('Guidance dimensions must be positive integers')
        if not 0 < self.residual_scale <= 1:
            raise ValueError('residual_scale must be in (0, 1]')
        if self.condition_source != 'human_question_frozen_input_embeddings':
            raise ValueError('Unsupported condition source')


class TextGuidedResidual(nn.Module):
    """Generate low-rank visual scale/shift from the task's question text."""
    def __init__(self, config: GuidanceConfig):
        super().__init__()
        self.config = config
        self.visual_norm = nn.LayerNorm(config.vision_dim)
        self.down = nn.Linear(config.vision_dim, config.rank)
        self.text_norm = nn.LayerNorm(config.text_dim)
        self.condition = nn.Sequential(nn.Linear(config.text_dim, config.rank), nn.GELU(),
                                       nn.Linear(config.rank, 2 * config.rank))
        self.up = nn.Linear(config.rank, config.vision_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden, text):
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.vision_dim:
            raise ValueError('Expected B x patches x vision_dim')
        if text.ndim != 2 or text.shape != (hidden.shape[0], self.config.text_dim):
            raise ValueError('Condition batch/text dimension mismatch')
        original = hidden
        hidden = hidden.to(dtype=self.down.weight.dtype)
        text = text.to(device=hidden.device, dtype=hidden.dtype)
        scale, shift = self.condition(self.text_norm(text)).chunk(2, dim=-1)
        rank_features = self.down(self.visual_norm(hidden))
        conditioned = rank_features * (1 + scale.tanh().unsqueeze(1)) + shift.tanh().unsqueeze(1)
        delta = self.config.residual_scale * self.up(torch.nn.functional.gelu(conditioned))
        return original + delta.to(dtype=original.dtype)


class GuidedEncoderBlock(nn.Module):
    """An actual structural replacement for one CLIPEncoderLayer.

    The original weights are unchanged. Only spatial tokens receive the
    residual. Explicit scoped conditions fail closed and prevent stale reuse.
    """
    def __init__(self, base_block, adapter):
        super().__init__()
        self.base_block = base_block
        self.adapter = adapter
        self._condition = None
        self._active = False
        self._enabled = True
        self.last_penalty = None

    @contextmanager
    def conditioned(self, text, *, enabled=True):
        if self._active:
            raise RuntimeError('Nested guidance contexts are unsupported')
        self._condition, self._active, self._enabled = text, True, enabled
        self.last_penalty = None
        try:
            yield
        finally:
            self._condition, self._active = None, False
            self.last_penalty = None

    def forward(self, *args, **kwargs):
        if not self._active:
            raise RuntimeError('Vision guidance requires an explicit question context')
        output = self.base_block(*args, **kwargs)
        if not self._enabled:
            return output
        if self._condition is None:
            raise ValueError('Missing question condition for image inference')
        hidden = output[0]
        spatial = hidden[:, 1:]
        adapted = self.adapter(spatial, self._condition)
        self.last_penalty = (adapted - spatial).float().square().mean() / (
            spatial.detach().float().square().mean() + 1e-6)
        return (torch.cat((hidden[:, :1], adapted), dim=1), *output[1:])


def install_guidance(model, config: GuidanceConfig):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if (base.config.vision_config.hidden_size != config.vision_dim or
            base.config.text_config.hidden_size != config.text_dim):
        raise ValueError('Guidance dimensions differ from the actual model')
    if base.config.vision_feature_layer != config.feature_layer:
        raise ValueError('Guidance must target the selected visual feature layer')
    if base.config.vision_feature_select_strategy != 'default':
        raise ValueError('v1 requires spatial-token selection (default)')
    layers = base.vision_tower.vision_model.encoder.layers
    # hidden_states[0] is embeddings; hidden_states[k+1] is block k's output.
    if not -(len(layers)+1) <= config.feature_layer <= len(layers):
        raise ValueError('Selected visual feature layer is out of range')
    hidden_index = list(range(len(layers) + 1))[config.feature_layer]
    block_index = hidden_index - 1
    if block_index < 0 or isinstance(layers[block_index], GuidedEncoderBlock):
        raise ValueError('Invalid/already guided visual block')
    reference = next(layers[block_index].parameters())
    model.requires_grad_(False)
    # FP32 master weights/Adam states for the small adapter even with BF16 LLaVA.
    adapter = TextGuidedResidual(config).to(device=reference.device)
    wrapped = GuidedEncoderBlock(layers[block_index], adapter)
    layers[block_index] = wrapped
    return wrapped, block_index


def condition_token_ids(tokenizer, prompt):
    """Only accept a prompt ending before the assistant completion."""
    if not isinstance(prompt, str) or not prompt.startswith('USER: ') or not prompt.endswith(' ASSISTANT:'):
        raise ValueError('Question condition must be a human-only inference prompt')
    if prompt.count('ASSISTANT:') != 1:
        raise ValueError('Assistant completions cannot enter the condition')
    human = prompt[len('USER: '):-len(' ASSISTANT:')].replace('<image>', '').strip()
    if not human:
        raise ValueError('Empty question condition')
    ids = tokenizer.encode(human, add_special_tokens=False)
    if not ids or len(ids) > 2048:
        raise ValueError('Question token length is empty or exceeds context budget')
    return ids


@torch.no_grad()
def question_conditions(model, tokenizer, image_records):
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    values = []
    for record in image_records:
        if record['modality'] != 'image':
            raise ValueError('Only image questions can condition the visual encoder')
        ids = torch.tensor(condition_token_ids(tokenizer, record['prompt']), device=device)
        values.append(embedding(ids).float().mean(0))
    return torch.stack(values)


def save_guidance(path, block, metadata):
    from safetensors.torch import save_file
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    tensors = {k: v.detach().cpu().contiguous() for k, v in block.adapter.state_dict().items()}
    save_file(tensors, destination / 'guidance.safetensors')
    checksum = hashlib.sha256((destination / 'guidance.safetensors').read_bytes()).hexdigest()
    payload = {'method': METHOD, 'config': asdict(block.adapter.config), 'weights_sha256': checksum,
               'metadata': metadata}
    (destination / 'guidance.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return payload


def load_guidance(path, block):
    from safetensors.torch import load_file
    source = Path(path)
    payload = json.loads((source / 'guidance.json').read_text())
    if payload['method'] != METHOD or payload['config'] != asdict(block.adapter.config):
        raise ValueError('Guidance architecture mismatch')
    if hashlib.sha256((source / 'guidance.safetensors').read_bytes()).hexdigest() != payload['weights_sha256']:
        raise ValueError('Guidance checkpoint SHA-256 mismatch')
    block.adapter.load_state_dict(load_file(source / 'guidance.safetensors'), strict=True)
    return payload


def verify_original_vision(model, integration):
    """Verify all original tensors despite the new encoder block's names."""
    from safetensors import safe_open
    from .moralclip_integration import sha
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    source = Path(integration['resolved_source']) / 'model.safetensors'
    if sha(source) != integration['weight_sha256']:
        raise ValueError('Original MoralCLIP source changed')
    checked = set()
    with safe_open(source, framework='pt', device='cpu') as stream:
        expected = {k for k in stream.keys() if k.startswith('vision_model.')}
        for name, tensor in base.vision_tower.state_dict().items():
            if '.adapter.' in name:
                continue
            original_name = name.replace('.base_block.', '.')
            if original_name not in expected or not torch.equal(
                    tensor.cpu(), stream.get_tensor(original_name).to(dtype=tensor.dtype)):
                raise ValueError('Frozen visual tensor changed: ' + name)
            checked.add(original_name)
    if checked != expected:
        raise ValueError('Incomplete original visual tensor verification')
    return len(checked)
