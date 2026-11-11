"""Verified original MoralCLIP vision tower for conditional-generation SFT.

The CLIP text encoder and global contrastive projection are deliberately not
inserted into LLaVA: the language decoder consumes spatial patch features.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_ID = 'anaaa2/moralclip-base'
REVISION = '1e337922befebde3f0cadb5749f37ae9521a0751'
WEIGHT_SHA256 = '4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a'
METHOD = 'original_moralclip_vision_polarity_sft_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resolve_source(source, revision=REVISION, local_files_only=True):
    path = Path(source).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(source, revision=revision,
                    local_files_only=local_files_only,
                    allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors']))
    return path.resolve()


def install_vision_tower(model, processor, source, *, expected_sha256=WEIGHT_SHA256,
                         revision=REVISION, local_files_only=True):
    """Strictly load every vision tensor, replace projector, fix image tokens."""
    import torch
    from safetensors import safe_open
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModel
    from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
    from .train import configure_processor

    path = resolve_source(source, revision, local_files_only)
    weight = path/'model.safetensors'
    checksum = sha(weight)
    if checksum != expected_sha256:
        raise ValueError('MoralCLIP weight SHA-256 mismatch; refusing vision replacement')
    config = CLIPConfig.from_pretrained(path, local_files_only=True).vision_config
    reference = next(model.multi_modal_projector.parameters())
    tower = CLIPVisionModel(config).to(dtype=reference.dtype, device=reference.device)
    with safe_open(weight, framework='pt', device='cpu') as stream:
        state = {key: stream.get_tensor(key) for key in stream.keys() if key.startswith('vision_model.')}
    # No randomly initialized or silently skipped visual layers are accepted.
    tower.load_state_dict(state, strict=True)
    for name, value in tower.state_dict().items():
        if not torch.equal(value.cpu(), state[name].to(dtype=value.dtype)):
            raise ValueError(f'MoralCLIP visual tensor mismatch: {name}')
    model.vision_tower = tower
    model.config.vision_config = tower.config
    model.config.vision_feature_layer = -2
    model.config.vision_feature_select_strategy = 'default'
    model.config.image_seq_length = (config.image_size//config.patch_size)**2
    model.multi_modal_projector = LlavaMultiModalProjector(model.config).to(
        device=reference.device, dtype=reference.dtype)
    processor.image_processor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
    tokens = configure_processor(processor, model.config, padding_side='right')
    tower.requires_grad_(False)
    tower.eval()
    return {'method': METHOD, 'model_id': MODEL_ID, 'revision': revision,
            'source': str(source), 'resolved_source': str(path), 'weight_sha256': checksum,
            'config_sha256': sha(path/'config.json'),
            'preprocessor_sha256': sha(path/'preprocessor_config.json'),
            'vision_tensors_verified': len(state), 'all_vision_tensors_match_source': True,
            'image_size': config.image_size, 'patch_size': config.patch_size,
            'vision_hidden_size': config.hidden_size, 'image_tokens': tokens,
            'feature_layer': -2, 'feature_strategy': 'default',
            'projector': [config.hidden_size, model.config.text_config.hidden_size,
                          model.config.text_config.hidden_size],
            'vision_trainable': False}


def configure_trainability(model, stage):
    """PEFT modules_to_save persists the dense projector in every checkpoint."""
    if stage not in ('projector', 'joint'):
        raise ValueError('Unknown MoralCLIP adaptation stage')
    for name, parameter in model.named_parameters():
        allowed = ('.multi_modal_projector.modules_to_save.' in name or
                   (stage == 'joint' and '.language_model.' in name and 'lora_' in name))
        parameter.requires_grad_(allowed)
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if not any('.multi_modal_projector.modules_to_save.' in name for name in names):
        raise ValueError('Dense projector must be registered as PEFT modules_to_save')
    if stage == 'joint' and not any('lora_' in name for name in names):
        raise ValueError('Joint polarity training requires language LoRA')
    if any(p.requires_grad for p in model.get_base_model().vision_tower.parameters()):
        raise ValueError('MoralCLIP visual tower must remain frozen for this baseline')
    return names


def verify_frozen_vision(model, integration):
    """Check every vision weight again after training/reloading."""
    import torch
    from safetensors import safe_open
    path = Path(integration['resolved_source'])/'model.safetensors'
    if sha(path) != integration['weight_sha256']:
        raise ValueError('Original MoralCLIP source changed')
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    with safe_open(path, framework='pt', device='cpu') as stream:
        for name, value in base.vision_tower.state_dict().items():
            if not torch.equal(value.cpu(), stream.get_tensor(name).to(dtype=value.dtype)):
                raise ValueError(f'Frozen MoralCLIP weight changed: {name}')
    return True


def validate_adapter_metadata(metadata, integration):
    original = metadata.get('moralclip_integration')
    if not original or original.get('method') != METHOD:
        raise ValueError('Initial adapter was not trained with the original MoralCLIP tower')
    for key in ('weight_sha256', 'config_sha256', 'preprocessor_sha256', 'image_tokens',
                'vision_hidden_size', 'feature_layer', 'projector'):
        if original.get(key) != integration.get(key):
            raise ValueError(f'Initial adapter MoralCLIP architecture/source mismatch: {key}')
