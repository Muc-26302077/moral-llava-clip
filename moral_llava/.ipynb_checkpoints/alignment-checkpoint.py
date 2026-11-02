"""Two-branch alignment on LLaVA-1.5, separate from the existing SFT baseline.

Images use the frozen vision tower, the multimodal projector and patch pooling.
Captions use the language decoder directly, without images or answer prompts.
The two trainable heads create a shared retrieval space. This is an adaptation
of LLaVA; it is not the original MoralCLIP training implementation.

Torch, Transformers and PEFT stay lazy so data preparation can run on a small
machine without installing the model runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "llava-hf/llava-1.5-7b-hf"
CHECKPOINT_VERSION = 1
_MODEL_CLASS: Any = None
LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)


@dataclass(frozen=True)
class AlignmentConfig:
    base_model_id: str = DEFAULT_BASE_MODEL
    revision: str | None = None
    vision_model_id: str | None = None
    vision_revision: str | None = None
    projection_dim: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    train_projector: bool = False
    temperature: float = 0.07
    train_logit_scale: bool = True
    vision_feature_layer: int = -2
    max_logit_scale: float = 100.0

    def __post_init__(self) -> None:
        if self.projection_dim <= 0 or self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("Projection size and LoRA rank/alpha must be positive.")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1).")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        if not math.isfinite(self.max_logit_scale) or self.max_logit_scale <= 0:
            raise ValueError("max_logit_scale must be positive and finite.")
        if 1 / self.temperature > self.max_logit_scale:
            raise ValueError("Initial inverse temperature exceeds max_logit_scale.")


def masked_mean_pool(hidden_states: Any, attention_mask: Any) -> Any:
    """Pool valid caption positions only; fail on an empty caption."""
    import torch

    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected hidden states [B,T,H] and attention mask [B,T].")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("Caption hidden states and attention mask have different shapes.")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("attention_mask must contain only 0 and 1.")
    weights = attention_mask.to(device=hidden_states.device, dtype=torch.float32)
    counts = weights.sum(dim=1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("A caption has no unmasked tokens.")
    return (hidden_states.float() * weights.unsqueeze(-1)).sum(dim=1) / counts


def _decoder(language_model: Any) -> Any:
    """Keep LoRA layers, but avoid allocating the full vocabulary logits."""
    base = (
        language_model.get_base_model()
        if hasattr(language_model, "get_base_model") else language_model
    )
    if hasattr(base, "get_decoder"):
        return base.get_decoder()
    if hasattr(base, "model"):
        return base.model
    raise ValueError("The language model does not expose a causal decoder.")


def _alignment_class() -> Any:
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    import torch
    from torch import nn
    from torch.nn import functional as F

    class AlignmentModel(nn.Module):
        def __init__(self, llava: Any, config: AlignmentConfig) -> None:
            super().__init__()
            self.llava = llava
            self.alignment_config = config
            width = int(llava.config.text_config.hidden_size)
            reference = next(llava.multi_modal_projector.parameters())
            self.image_projection = nn.Linear(width, config.projection_dim, bias=False)
            self.text_projection = nn.Linear(width, config.projection_dim, bias=False)
            self.image_projection.to(device=reference.device, dtype=reference.dtype)
            self.text_projection.to(device=reference.device, dtype=reference.dtype)
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1 / config.temperature), dtype=torch.float32,
                             device=reference.device),
                requires_grad=config.train_logit_scale,
            )

        @property
        def device(self) -> Any:
            return self.image_projection.weight.device

        @property
        def trainable_parameter_names(self) -> list[str]:
            return [name for name, value in self.named_parameters() if value.requires_grad]

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            # 视觉塔始终冻结，别让训练模式改了它的 dropout 行为。
            self.llava.vision_tower.eval()
            return self

        def encode_images(self, pixel_values: Any) -> Any:
            if pixel_values.ndim != 4:
                raise ValueError("pixel_values must be a [B,C,H,W] tensor.")
            tower = self.llava.vision_tower
            reference = next(tower.parameters())
            pixel_values = pixel_values.to(device=reference.device, dtype=reference.dtype)
            with torch.no_grad():
                output = tower(pixel_values, output_hidden_states=True, return_dict=True)
                selected = output.hidden_states[self.alignment_config.vision_feature_layer]
                strategy = self.llava.config.vision_feature_select_strategy
                if strategy == "default":
                    selected = selected[:, 1:]
                elif strategy != "full":
                    raise ValueError(f"Unsupported vision feature selection: {strategy}")
            expected = int(self.llava.config.image_seq_length)
            if selected.shape[1] != expected:
                raise ValueError(
                    f"Vision tower produced {selected.shape[1]} image tokens, expected {expected}."
                )
            # pooling 放在 projector 之后，梯度才能训练新接的 projector。
            projected = self.llava.multi_modal_projector(selected)
            pooled = projected.float().mean(dim=1)
            embedding = self.image_projection(pooled.to(self.image_projection.weight.dtype))
            return F.normalize(embedding.float(), dim=-1, eps=1e-12)

        def encode_texts(self, input_ids: Any, attention_mask: Any) -> Any:
            if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
                raise ValueError("input_ids and attention_mask must have shape [B,T].")
            if torch.any(input_ids == self.llava.config.image_token_index):
                raise ValueError("The caption branch accepts pure text, without <image> tokens.")
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            output = _decoder(self.llava.language_model)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            pooled = masked_mean_pool(output.last_hidden_state, attention_mask)
            embedding = self.text_projection(pooled.to(self.text_projection.weight.dtype))
            return F.normalize(embedding.float(), dim=-1, eps=1e-12)

        def forward(self, pixel_values: Any, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
            image_embeddings = self.encode_images(pixel_values)
            text_embeddings = self.encode_texts(input_ids, attention_mask)
            scale = self.logit_scale.float().clamp(max=math.log(
                self.alignment_config.max_logit_scale
            )).exp()
            logits = scale * image_embeddings @ text_embeddings.T
            return {
                "image_embeddings": image_embeddings,
                "text_embeddings": text_embeddings,
                "logits_per_image": logits,
                "logits_per_text": logits.T,
                "logit_scale": scale,
            }

        def forward_sft(self, **inputs: Any) -> Any:
            """Optional assistant-only SFT; callers supply the masked labels."""
            return self.llava(**inputs)

    _MODEL_CLASS = AlignmentModel
    return _MODEL_CLASS


def _set_trainable(model: Any, trainable: bool) -> None:
    config = model.alignment_config
    for name, parameter in model.named_parameters():
        allowed = (
            (name.startswith("llava.language_model.") and "lora_" in name)
            or (name.startswith("llava.multi_modal_projector.") and config.train_projector)
            or name.startswith("image_projection.")
            or name.startswith("text_projection.")
            or (name == "logit_scale" and config.train_logit_scale)
        )
        parameter.requires_grad_(bool(trainable and allowed))
    if trainable and not model.trainable_parameter_names:
        raise ValueError("The alignment model has no trainable parameters.")


def build_alignment_model(llava: Any, config: AlignmentConfig | None = None, *,
                          trainable: bool = True, attach_lora: bool = True) -> Any:
    """Wrap an already loaded LLaVA; useful for tiny random-config checks too.

    ``attach_lora=False`` is for a language model whose adapter was loaded by
    the checkpoint loader. It does not load a baseline SFT adapter implicitly.
    """
    config = config or AlignmentConfig()
    if getattr(llava.config, "model_type", None) != "llava":
        raise ValueError("This alignment module supports LLaVA-1.5, model_type='llava'.")
    for parameter in llava.parameters():
        parameter.requires_grad_(False)
    if attach_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        llava.language_model = get_peft_model(llava.language_model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, bias="none",
            target_modules=list(LORA_TARGETS),
            inference_mode=not trainable,
        ))
    llava.config.vision_feature_layer = config.vision_feature_layer
    llava.config.use_cache = False
    llava.language_model.config.use_cache = False
    model = _alignment_class()(llava, config)
    _set_trainable(model, trainable)
    return model


def _resolve_dtype(dtype: Any) -> Any:
    import torch

    if dtype is None or dtype == "auto":
        return torch.float32
    if isinstance(dtype, str):
        options = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32,
        }
        if dtype not in options:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return options[dtype]
    return dtype


def _prepare_processor(processor: Any, llava: Any) -> None:
    from moral_llava.train import configure_processor

    crop = processor.image_processor.crop_size
    height, width = int(crop["height"]), int(crop["width"])
    patch = int(llava.config.vision_config.patch_size)
    if height % patch or width % patch:
        raise ValueError("Processor dimensions must be divisible by the vision patch size.")
    llava.config.image_seq_length = (height // patch) * (width // patch)
    if llava.config.vision_feature_select_strategy == "full":
        llava.config.image_seq_length += 1
    configure_processor(processor, llava.config, padding_side="right")


def load_alignment_model(
    base_model_id: str = DEFAULT_BASE_MODEL,
    *,
    revision: str | None = None,
    vision_model_id: str | None = None,
    vision_revision: str | None = None,
    projection_dim: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    train_projector: bool | None = None,
    temperature: float = 0.07,
    train_logit_scale: bool = True,
    vision_feature_layer: int = -2,
    max_logit_scale: float = 100.0,
    dtype: Any = None,
    device: Any = None,
    local_files_only: bool = False,
    trainable: bool = True,
    gradient_checkpointing: bool = False,
    adapter_path: str | Path | None = None,
    processor_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Create a new experiment; returning ``(model, LlavaProcessor)``.

    Providing a CLIP ViT-B/16 source replaces the visual tower and creates
    a fresh 768 -> 4096 -> 4096 projector. It needs training before use.
    ``vision_model_id=None`` retains the original LLaVA tower/projector.
    """
    from transformers import AutoProcessor, CLIPImageProcessor, CLIPVisionModel
    from transformers import LlavaForConditionalGeneration
    from transformers.models.llava.modeling_llava import LlavaMultiModalProjector

    dtype = _resolve_dtype(dtype)
    config = AlignmentConfig(
        base_model_id=base_model_id, revision=revision,
        vision_model_id=vision_model_id, vision_revision=vision_revision,
        projection_dim=projection_dim, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        train_projector=(vision_model_id is not None) if train_projector is None else train_projector,
        temperature=temperature, train_logit_scale=train_logit_scale,
        vision_feature_layer=vision_feature_layer, max_logit_scale=max_logit_scale,
    )
    base_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if revision:
        base_kwargs["revision"] = revision
    processor_kwargs = {"local_files_only": local_files_only}
    if not processor_path and revision:
        processor_kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(str(processor_path or base_model_id), **processor_kwargs)
    llava = LlavaForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=dtype, attn_implementation="sdpa",
        low_cpu_mem_usage=True, **base_kwargs,
    )
    if vision_model_id:
        vision_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if vision_revision:
            vision_kwargs["revision"] = vision_revision
        llava.vision_tower = CLIPVisionModel.from_pretrained(
            vision_model_id, torch_dtype=dtype, attn_implementation="sdpa", **vision_kwargs
        )
        llava.config.vision_config = llava.vision_tower.config
        llava.config.vision_feature_select_strategy = "default"
        llava.multi_modal_projector = LlavaMultiModalProjector(llava.config).to(dtype=dtype)
        # B/16 用 224 裁剪，不能沿用原模型 336 的 processor。
        processor.image_processor = CLIPImageProcessor.from_pretrained(vision_model_id, **vision_kwargs)
    _prepare_processor(processor, llava)
    if adapter_path is not None:
        from peft import PeftModel

        llava.language_model = PeftModel.from_pretrained(
            llava.language_model, str(adapter_path), is_trainable=trainable,
            local_files_only=local_files_only,
        )
    model = build_alignment_model(
        llava, config, trainable=trainable, attach_lora=adapter_path is None
    )
    if device is not None:
        model.to(device)
    if gradient_checkpointing and trainable:
        model.llava.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.llava.language_model.enable_input_require_grads()
    model.train(trainable)
    return model, processor


def save_alignment_checkpoint(
    model: Any, processor: Any, checkpoint: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save adapters and alignment components, without duplicating the 7B base."""
    import torch

    path = Path(checkpoint).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if not hasattr(model.llava.language_model, "peft_config"):
        raise ValueError("The language branch must contain a PEFT LoRA adapter.")
    # embedding 没训练也没扩词表，别让 PEFT 保存时联网查 Vicuna 配置。
    model.llava.language_model.save_pretrained(
        path / "language_adapter", safe_serialization=True, save_embedding_layers=False,
    )
    processor.save_pretrained(path / "processor")
    dense = {
        "image_projection": model.image_projection.state_dict(),
        "text_projection": model.text_projection.state_dict(),
        "logit_scale": model.logit_scale.detach().cpu(),
    }
    if model.alignment_config.train_projector or model.alignment_config.vision_model_id:
        dense["multi_modal_projector"] = model.llava.multi_modal_projector.state_dict()
    dense = {
        key: {name: value.detach().cpu() for name, value in state.items()}
        if isinstance(state, dict) else state
        for key, state in dense.items()
    }
    torch.save(dense, path / "alignment_components.pt")
    record = {
        "format_version": CHECKPOINT_VERSION,
        "method": "llava_two_branch_alignment",
        "config": asdict(model.alignment_config),
        "expected_image_tokens": int(model.llava.config.image_seq_length),
        "vision_hidden_size": int(model.llava.config.vision_config.hidden_size),
        "text_hidden_size": int(model.llava.config.text_config.hidden_size),
        "metadata": metadata or {},
        "scope": "LLaVA adaptation; not a complete reproduction of the MoralCLIP paper",
    }
    (path / "alignment_config.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_alignment_checkpoint(
    checkpoint: str | Path, *, device: Any = None, dtype: Any = None,
    local_files_only: bool = True, trainable: bool = False,
    gradient_checkpointing: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """Restore the actual retrieval model; returns ``(model, processor, metadata)``."""
    import torch

    path = Path(checkpoint).expanduser().resolve()
    record = json.loads((path / "alignment_config.json").read_text(encoding="utf-8"))
    if record.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported alignment checkpoint version.")
    config = AlignmentConfig(**record["config"])
    model, processor = load_alignment_model(
        **asdict(config), dtype=dtype, device=device,
        local_files_only=local_files_only, trainable=trainable,
        gradient_checkpointing=gradient_checkpointing,
        adapter_path=path / "language_adapter", processor_path=path / "processor",
    )
    dense = torch.load(path / "alignment_components.pt", map_location="cpu", weights_only=True)
    required = {"image_projection", "text_projection", "logit_scale"}
    if config.train_projector or config.vision_model_id:
        required.add("multi_modal_projector")
    if set(dense) != required:
        raise ValueError(f"Alignment component mismatch: expected {sorted(required)}, got {sorted(dense)}")
    model.image_projection.load_state_dict(dense["image_projection"], strict=True)
    model.text_projection.load_state_dict(dense["text_projection"], strict=True)
    if "multi_modal_projector" in dense:
        model.llava.multi_modal_projector.load_state_dict(dense["multi_modal_projector"], strict=True)
    with torch.no_grad():
        model.logit_scale.copy_(dense["logit_scale"])
    if int(model.llava.config.image_seq_length) != int(record["expected_image_tokens"]):
        raise ValueError("Restored processor image-token count does not match checkpoint.")
    _set_trainable(model, trainable)
    model.train(trainable)
    return model, processor, record
