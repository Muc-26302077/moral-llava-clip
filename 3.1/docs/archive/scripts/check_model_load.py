"""Offline BF16 LLaVA loading and one-image forward check on a CUDA GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moral_llava.train import (
    DEFAULT_MODEL_ID,
    LlavaSFTCollator,
    configure_processor,
    context_length,
    find_language_lora_targets,
    load_records,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--train-data", default="data/processed/train.json")
    parser.add_argument("--image-root", default="data/raw/SMID_moralclip")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser


def run_check(args: argparse.Namespace) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("This check requires an NVIDIA CUDA GPU.")
    if not 0 <= args.gpu_index < torch.cuda.device_count():
        raise ValueError("gpu-index is outside the available CUDA devices.")
    if args.max_length <= 0 or args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("max-length, lora-r and lora-alpha must be positive.")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora-dropout must be in [0, 1).")
    torch.cuda.set_device(args.gpu_index)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA GPU does not support bfloat16.")
    device = torch.device("cuda", args.gpu_index)
    records = load_records(args.train_data, args.image_root)
    if not 0 <= args.sample_index < len(records):
        raise ValueError(f"sample-index must be between 0 and {len(records) - 1}.")
    record = records[args.sample_index]

    # 只读已有缓存，模型没下全就报错，不在这里补下载。
    pretrained_kwargs = {"local_files_only": True}
    if args.revision:
        pretrained_kwargs["revision"] = args.revision
    torch.cuda.reset_peak_memory_stats(device)
    processor = AutoProcessor.from_pretrained(args.model_id, **pretrained_kwargs)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        **pretrained_kwargs,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": args.gpu_index},
    )
    expected_image_tokens = configure_processor(processor, model.config, padding_side="right")
    if expected_image_tokens != 576:
        raise ValueError(f"Expected 576 LLaVA-1.5 image tokens, got {expected_image_tokens}.")
    if args.max_length > context_length(model.config):
        raise ValueError("max-length exceeds the model's context limit.")
    model.config.use_cache = False

    # 和训练脚本一致：底座全冻结，只给语言层挂 LoRA。
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = find_language_lora_targets(model, torch.nn.Linear)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=targets,
    ))
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(
        ".language_model." not in name or "lora_" not in name for name, _ in trainable
    ):
        raise AssertionError("Only language_model LoRA parameters may be trainable.")
    lora_parameters = sum(parameter.numel() for _, parameter in trainable)
    collator = LlavaSFTCollator(
        processor,
        image_token_id=model.config.image_token_index,
        expected_image_tokens=expected_image_tokens,
        max_length=args.max_length,
    )
    inputs = collator([record])
    image_tokens = int((inputs["input_ids"] == model.config.image_token_index).sum().item())
    if image_tokens != 576:
        raise AssertionError(f"The processed sample has {image_tokens} image tokens, expected 576.")
    input_length = int(inputs["attention_mask"].sum().item())
    supervised_tokens = int((inputs["labels"] != -100).sum().item())
    inputs = {
        key: value.to(device=device, dtype=torch.bfloat16) if value.is_floating_point()
        else value.to(device=device)
        for key, value in inputs.items()
    }

    # 只看一次前向是否正常，不反传、不生成答案，也不保存权重。
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**inputs)
    if outputs.loss is None or not torch.isfinite(outputs.loss).item():
        raise AssertionError("The one-image forward pass did not return a finite loss.")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Unexpected parameter gradients after a no_grad forward pass.")
    torch.cuda.synchronize(device)
    return {
        "status": "PASS",
        "model_id": args.model_id,
        "sample_id": record["id"],
        "gpu": torch.cuda.get_device_name(device),
        "dtype": "bfloat16",
        "attention": "sdpa",
        "input_length": input_length,
        "image_tokens": image_tokens,
        "supervised_tokens": supervised_tokens,
        "lora_target_modules": len(targets),
        "lora_trainable_parameters": lora_parameters,
        "loss": float(outputs.loss.item()),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 1024 ** 3, 3),
        "local_files_only": True,
        "parameter_updates": 0,
    }


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        result = run_check(args)
    except (ImportError, ValueError, RuntimeError, OSError, AssertionError) as error:
        parser.exit(2, f"Model forward check failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
