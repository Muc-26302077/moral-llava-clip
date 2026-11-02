"""LoRA/QLoRA SFT for the Hugging Face LLaVA-1.5 implementation.

Heavy dependencies are imported only when running training or the collator.
The sequence-building helpers are deliberately plain Python so masking can be
verified without downloading a model or importing PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence


DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IGNORE_INDEX = -100
LORA_LAYER_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def build_prompt(human_text: str) -> str:
    """Use the same LLaVA-1.5 inference prefix during training and inference."""
    if not isinstance(human_text, str) or human_text.count("<image>") != 1:
        raise ValueError("Each human message must contain exactly one <image> token.")
    return f"USER: {human_text.strip()} ASSISTANT:"


def unpad_input_ids(
    input_ids: Sequence[int], attention_mask: Sequence[int]
) -> list[int]:
    """Remove processor padding by position, never by the value of a pad token."""
    if len(input_ids) != len(attention_mask):
        raise ValueError("input_ids and attention_mask must have equal lengths.")
    if any(value not in (0, 1) for value in attention_mask):
        raise ValueError("attention_mask must contain only 0 or 1.")
    # 别按 token 值去掉 padding，EOS 可能和它共用一个 id。
    result = [int(token) for token, keep in zip(input_ids, attention_mask) if keep]
    if not result:
        raise ValueError("An empty prompt is not supported.")
    return result


def build_supervised_sequence(
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    *,
    eos_token_id: int,
    image_token_id: int,
    expected_image_tokens: int,
    max_length: int,
) -> dict[str, list[int]]:
    """Append an answer and one EOS; supervise only those appended tokens.

    ``answer_ids`` must be tokenized without special tokens and without EOS.
    Oversize samples are rejected, rather than truncating expanded image tokens
    or silently discarding some of the structured answer.
    """
    prompt = [int(token) for token in prompt_ids]
    answer = [int(token) for token in answer_ids]
    if not prompt or not answer:
        raise ValueError("Prompt and answer must both contain tokens.")
    if expected_image_tokens <= 0 or max_length <= 0:
        raise ValueError("expected_image_tokens and max_length must be positive.")
    image_count = prompt.count(image_token_id)
    if image_count != expected_image_tokens:
        raise ValueError(
            f"Expected {expected_image_tokens} expanded image tokens, got {image_count}. "
            "Check processor.patch_size, num_additional_image_tokens and "
            "vision_feature_select_strategy."
        )
    if image_token_id in answer:
        raise ValueError("The structured answer cannot contain image tokens.")
    if eos_token_id in answer:
        raise ValueError("answer_ids must not already contain EOS.")
    # EOS 也要算 loss，让模型学会在 JSON 结束后停下来。
    completion = answer + [eos_token_id]
    input_ids = prompt + completion
    if len(input_ids) > max_length:
        raise ValueError(
            f"Sample needs {len(input_ids)} tokens including image tokens and EOS, "
            f"but max_length={max_length}. Increase max_length within the model's "
            "context limit; this pipeline never truncates samples."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt) + completion,
    }


def pad_supervised_batch(
    examples: Sequence[dict[str, Sequence[int]]],
    *,
    pad_token_id: int,
    padding_side: str = "right",
) -> dict[str, list[list[int]]]:
    """Pad by position so a supervised EOS survives even when pad_id == eos_id."""
    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    if padding_side not in ("left", "right"):
        raise ValueError("padding_side must be 'left' or 'right'.")
    width = max(len(example["input_ids"]) for example in examples)
    result: dict[str, list[list[int]]] = {
        "input_ids": [], "attention_mask": [], "labels": []
    }
    for example in examples:
        ids = list(example["input_ids"])
        attention = list(example["attention_mask"])
        labels = list(example["labels"])
        if not ids or len(ids) != len(attention) or len(ids) != len(labels):
            raise ValueError("Each example must have equally sized, nonempty fields.")
        if any(value not in (0, 1) for value in attention):
            raise ValueError("attention_mask must contain only 0 or 1.")
        labels = [label if keep else IGNORE_INDEX for label, keep in zip(labels, attention)]
        missing = width - len(ids)
        pads = {
            "input_ids": [pad_token_id] * missing,
            "attention_mask": [0] * missing,
            "labels": [IGNORE_INDEX] * missing,
        }
        for key, values in (("input_ids", ids), ("attention_mask", attention), ("labels", labels)):
            result[key].append(
                values + pads[key] if padding_side == "right" else pads[key] + values
            )
    return result


def safe_image_path(image_root: str | Path, relative_path: str) -> Path:
    """Resolve a manifest image strictly inside its image root."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("The image field must be a nonempty relative path.")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image path: {relative_path!r}")
    root = Path(image_root).expanduser().resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Image path escapes image_root: {relative_path!r}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    return resolved


def load_records(
    data_path: str | Path,
    image_root: str | Path,
    *,
    require_targets: bool = True,
) -> list[dict[str, Any]]:
    """Read standard two-turn LLaVA records and normalize their training fields."""
    from moral_llava.schema import parse_prediction

    with Path(data_path).open(encoding="utf-8") as handle:
        raw_records = json.load(handle)
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"{data_path} must contain a nonempty JSON array.")
    normalized = []
    seen: set[str] = set()
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object.")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError(f"Record {index} has a missing, invalid or duplicate id.")
        seen.add(record_id)
        turns = record.get("conversations")
        if (
            not isinstance(turns, list)
            or not turns
            or not isinstance(turns[0], dict)
            or turns[0].get("from") != "human"
        ):
            raise ValueError(f"{record_id}: the first conversation turn must be human.")
        human = turns[0].get("value")
        prompt = build_prompt(human)
        target_json = None
        if require_targets:
            if (
                len(turns) != 2
                or not isinstance(turns[1], dict)
                or turns[1].get("from") != "gpt"
                or not isinstance(turns[1].get("value"), str)
            ):
                raise ValueError(f"{record_id}: expected exactly one human/gpt exchange.")
            target = parse_prediction(turns[1]["value"])
            target_json = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
        normalized.append({
            "id": record_id,
            "image_path": str(safe_image_path(image_root, record.get("image"))),
            "prompt": prompt,
            "target_json": target_json,
        })
    return normalized


def configure_processor(processor: Any, config: Any, *, padding_side: str) -> int:
    """Restore attributes needed by Transformers 4.49's image-token expansion."""
    if config.model_type != "llava":
        raise ValueError("This pipeline supports plain LLaVA-1.5 (model_type='llava').")
    vision = config.vision_config
    patch_size = int(vision.patch_size)
    processor.patch_size = patch_size
    # CLIP 多一个 CLS；default 会去掉它，这里要对上。
    processor.num_additional_image_tokens = 1
    strategy = config.vision_feature_select_strategy
    if strategy not in ("default", "full"):
        raise ValueError(f"Unsupported vision feature strategy: {strategy}")
    processor.vision_feature_select_strategy = strategy
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = False
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    if image_token_id != config.image_token_index:
        raise ValueError("Processor image-token id does not match model.config.image_token_index.")
    crop = getattr(processor.image_processor, "crop_size", None)
    if isinstance(crop, dict) and "height" in crop and "width" in crop:
        height, width = int(crop["height"]), int(crop["width"])
    else:
        height = width = int(vision.image_size)
    if height % patch_size or width % patch_size:
        raise ValueError("Processed image dimensions must be divisible by patch_size.")
    count = (height // patch_size) * (width // patch_size) + 1
    if strategy == "default":
        count -= 1
    configured_count = getattr(config, "image_seq_length", count)
    if count != configured_count:
        raise ValueError(
            f"Processor would create {count} image tokens; model expects {configured_count}."
        )
    return count


def context_length(config: Any) -> int:
    return int(config.text_config.max_position_embeddings)


class ConversationDataset:
    """Map-style dataset; image decoding remains lazy inside the collator."""

    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class LlavaSFTCollator:
    def __init__(
        self,
        processor: Any,
        *,
        image_token_id: int,
        expected_image_tokens: int,
        max_length: int,
    ) -> None:
        self.processor = processor
        self.image_token_id = image_token_id
        self.expected_image_tokens = expected_image_tokens
        self.max_length = max_length

    def __call__(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from PIL import Image, ImageOps

        images = []
        for record in records:
            with Image.open(record["image_path"]) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        # 先去掉 prompt 的 padding 再拼答案，短样本才不会错位。
        processed = self.processor(
            text=[record["prompt"] for record in records],
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        examples = []
        for index, record in enumerate(records):
            prompt_ids = unpad_input_ids(
                processed["input_ids"][index].tolist(),
                processed["attention_mask"][index].tolist(),
            )
            answer_ids = self.processor.tokenizer.encode(
                " " + record["target_json"], add_special_tokens=False
            )
            try:
                examples.append(build_supervised_sequence(
                    prompt_ids,
                    answer_ids,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    image_token_id=self.image_token_id,
                    expected_image_tokens=self.expected_image_tokens,
                    max_length=self.max_length,
                ))
            except ValueError as error:
                raise ValueError(f"{record['id']}: {error}") from error
        padded = pad_supervised_batch(
            examples,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        result = {
            key: torch.tensor(value, dtype=torch.long) for key, value in padded.items()
        }
        result["pixel_values"] = processed["pixel_values"]
        return result


def find_language_lora_targets(model: Any, linear_type: type) -> list[str]:
    """Use full module names to avoid matching vision/projector linear layers."""
    targets = [
        name for name, module in model.named_modules()
        if name.startswith("language_model.")
        and name.rsplit(".", 1)[-1] in LORA_LAYER_NAMES
        and isinstance(module, linear_type)
    ]
    if not targets:
        raise ValueError("No supported language_model linear layers were found for LoRA.")
    return sorted(targets)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", help="Optional Hugging Face model revision/commit.")
    parser.add_argument("--train-data", default="data/processed/train.json")
    parser.add_argument("--validation-data", default="data/processed/validation.json")
    parser.add_argument("--image-root", default="data/raw/SMID_moralclip")
    parser.add_argument("--output-dir", default="runs/moral-llava-1.5")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable CUDA QLoRA (NF4).")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value for a bounded GPU smoke run.")
    parser.add_argument("--resume-from-checkpoint", help="Trainer checkpoint directory to resume.")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps",
        "learning_rate", "max_length", "lora_r", "lora_alpha", "logging_steps",
        "save_total_limit",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.num_workers < 0 or args.weight_decay < 0:
        raise ValueError("num_workers and weight_decay cannot be negative.")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.lora_dropout < 1:
        raise ValueError("warmup_ratio and lora_dropout must be in [0, 1).")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or positive.")
    if Path(args.train_data).resolve() == Path(args.validation_data).resolve():
        raise ValueError("Training and validation must be separate files.")


def run_training(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training requires an NVIDIA CUDA GPU. This Mac/CPU environment can "
            "prepare data and run the pure-Python checks, but cannot run this training command."
        )
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration,
        Trainer, TrainerCallback, TrainingArguments, set_seed,
    )
    import accelerate
    import peft
    import transformers

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    bf16_supported = torch.cuda.is_bf16_supported()
    if args.precision == "bf16" and not bf16_supported:
        raise RuntimeError("This CUDA device does not support bfloat16; choose --precision fp16.")
    use_bf16 = bf16_supported if args.precision == "auto" else args.precision == "bf16"
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("output-dir is nonempty. Choose a new directory or pass --resume-from-checkpoint.")
    train_records = load_records(args.train_data, args.image_root)
    validation_records = load_records(args.validation_data, args.image_root)
    train_ids = {record["id"] for record in train_records}
    validation_ids = {record["id"] for record in validation_records}
    if train_ids & validation_ids:
        raise ValueError("Training and validation share image ids.")
    train_images = {record["image_path"] for record in train_records}
    if train_images & {record["image_path"] for record in validation_records}:
        raise ValueError("Training and validation share image files.")
    set_seed(args.seed)
    pretrained_kwargs = {"local_files_only": args.local_files_only}
    if args.revision:
        pretrained_kwargs["revision"] = args.revision
    processor = AutoProcessor.from_pretrained(args.model_id, **pretrained_kwargs)
    model_kwargs = {
        **pretrained_kwargs,
        "torch_dtype": compute_dtype,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
        )
        model_kwargs["device_map"] = {"": local_rank}
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    expected_image_tokens = configure_processor(processor, model.config, padding_side="right")
    if args.max_length > context_length(model.config):
        raise ValueError(f"max_length exceeds the model context of {context_length(model.config)} tokens.")
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    targets = find_language_lora_targets(model, torch.nn.Linear)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=targets,
    ))
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or any(
        ".language_model." not in name or "lora_" not in name for name in trainable_names
    ):
        raise RuntimeError("Freeze check failed: only language_model LoRA weights may be trainable.")
    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    collator = LlavaSFTCollator(
        processor,
        image_token_id=model.config.image_token_index,
        expected_image_tokens=expected_image_tokens,
        max_length=args.max_length,
    )
    metadata = {
        "format_version": 1,
        "base_model_id": args.model_id,
        "base_model_revision": args.revision,
        "resolved_base_commit": getattr(model.config, "_commit_hash", None),
        "task": "image_only_moral_multilabel_sft",
        "prompt_format": "USER: <image>\\n{TASK_PROMPT} ASSISTANT:",
        "supervision": "assistant structured JSON plus EOS; prompt/image/padding masked",
        "missing_foundation_value": "not_annotated",
        "freeze_policy": "vision tower, multimodal projector, base language weights frozen",
        "lora_target_modules": targets,
        "expected_image_tokens": expected_image_tokens,
        "processor": {
            "patch_size": processor.patch_size,
            "num_additional_image_tokens": processor.num_additional_image_tokens,
            "vision_feature_select_strategy": processor.vision_feature_select_strategy,
        },
        "training_examples": len(train_records),
        "validation_examples": len(validation_records),
        "selection_metric": "validation loss (minimum)",
        "precision": "bf16" if use_bf16 else "fp16",
        "load_in_4bit": args.load_in_4bit,
        "arguments": vars(args),
        "versions": {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "accelerate": accelerate.__version__,
        },
    }

    def save_metadata(directory: Path, state: Any) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        saved = {
            **metadata,
            "global_step": state.global_step,
            "best_model_checkpoint": state.best_model_checkpoint,
            "best_validation_loss": state.best_metric,
        }
        (directory / "training_metadata.json").write_text(
            json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        processor.save_pretrained(directory)

    class MetadataCallback(TrainerCallback):
        def on_save(self, training_args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if training_args.should_save:
                save_metadata(output_dir / f"checkpoint-{state.global_step}", state)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ConversationDataset(train_records),
        eval_dataset=ConversationDataset(validation_records),
        data_collator=collator,
        processing_class=processor,
        callbacks=[MetadataCallback()],
    )
    # LLaVA 自己算平均 loss，别让 Trainer 误以为它处理了梯度累积的归一化。
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # 这里保存的是验证集 loss 最低的 adapter，不一定是最后一轮。
    trainer.save_model(str(output_dir))
    trainer.save_state()
    if trainer.is_world_process_zero():
        save_metadata(output_dir, trainer.state)
        print(f"Saved validation-selected LoRA adapter and processor to {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        run_training(args)
    except ModuleNotFoundError as error:
        parser.exit(2, f"Missing training dependency: {error.name}. Install requirements-train.txt on the CUDA server.\n")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Training error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
