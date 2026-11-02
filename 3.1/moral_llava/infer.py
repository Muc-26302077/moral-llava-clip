"""Image-only structured moral prediction with base or LoRA LLaVA-1.5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from moral_llava.train import (
    DEFAULT_MODEL_ID, build_prompt, configure_processor, context_length,
    load_records, unpad_input_ids,
)


def format_prediction(record_id: str, raw_prediction: str) -> dict[str, Any]:
    """Retain every raw response and expose every parse failure explicitly."""
    from moral_llava.schema import parse_prediction, render_answer

    result: dict[str, Any] = {
        "id": record_id,
        "raw_prediction": raw_prediction,
        "prediction": None,
        "parse_error": None,
        "answer": None,
    }
    try:
        prediction = parse_prediction(raw_prediction)
    except ValueError as error:
        # 解析失败也保留原文，后面统计时别把这些样本漏掉。
        result["parse_error"] = str(error)
    else:
        result["prediction"] = prediction
        result["answer"] = render_answer(prediction)
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="LLaVA JSON array for batch prediction; labels are never fed to the model.")
    source.add_argument("--image", help="A single local image.")
    parser.add_argument("--image-root", default="data/raw/SMID_moralclip")
    parser.add_argument("--adapter", help="Local LoRA output directory or Trainer checkpoint; omit for zero-shot.")
    parser.add_argument("--model-id", default=None, help=f"Base model (default: adapter metadata or {DEFAULT_MODEL_ID}).")
    parser.add_argument("--revision", help="Optional base-model revision; otherwise use adapter metadata.")
    parser.add_argument("--output", default="runs/predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum prompt plus reserved generation length; never truncate.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="CUDA only; use the same setting for baseline comparisons.")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.max_length <= 0:
        raise ValueError("batch-size, max-new-tokens and max-length must be positive.")
    if args.max_new_tokens >= args.max_length:
        raise ValueError("max-new-tokens must be smaller than max-length.")
    if args.adapter and not Path(args.adapter).expanduser().is_dir():
        raise ValueError("--adapter must be a local adapter/checkpoint directory.")
    if args.image and not Path(args.image).expanduser().is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")


def run_inference(args: argparse.Namespace) -> None:
    from moral_llava.schema import TASK_PROMPT

    # 先检查数据和图片路径，免得加载完大模型才发现文件不对。
    if args.data:
        records = load_records(args.data, args.image_root, require_targets=False)
    else:
        image_path = Path(args.image).expanduser().resolve()
        records = [{
            "id": image_path.stem,
            "image_path": str(image_path),
            "prompt": build_prompt("<image>\n" + TASK_PROMPT),
        }]
    adapter_dir = Path(args.adapter).expanduser().resolve() if args.adapter else None
    metadata: dict[str, Any] = {}
    if adapter_dir:
        metadata_path = adapter_dir / "training_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_id = args.model_id or metadata.get("base_model_id") or DEFAULT_MODEL_ID
    revision = (
        args.revision or metadata.get("resolved_base_commit")
        or metadata.get("base_model_revision")
    )
    if args.model_id and metadata.get("base_model_id") not in (None, args.model_id):
        raise ValueError("--model-id conflicts with the adapter's recorded base model.")

    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if args.load_in_4bit and device != "cuda":
        raise ValueError("4-bit bitsandbytes inference requires an NVIDIA CUDA GPU.")
    if args.precision == "auto":
        dtype = (
            torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16 if device in ("cuda", "mps") else torch.float32
        )
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    if dtype == torch.bfloat16 and (
        device == "mps" or (device == "cuda" and not torch.cuda.is_bf16_supported())
    ):
        raise ValueError("This device does not support the requested bf16 mode.")
    pretrained_kwargs = {"local_files_only": args.local_files_only}
    if revision:
        pretrained_kwargs["revision"] = revision
    processor_source = (
        str(adapter_dir) if adapter_dir and (adapter_dir / "processor_config.json").is_file()
        else model_id
    )
    processor_kwargs = {"local_files_only": args.local_files_only}
    if processor_source == model_id and revision:
        processor_kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)
    model_kwargs = {
        **pretrained_kwargs,
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
        )
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    expected_image_tokens = configure_processor(processor, model.config, padding_side="left")
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    maximum_context = min(args.max_length, context_length(model.config))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    parse_failures = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for offset in range(0, len(records), args.batch_size):
                current = records[offset:offset + args.batch_size]
                images = []
                for record in current:
                    with Image.open(record["image_path"]) as image:
                        images.append(ImageOps.exif_transpose(image).convert("RGB"))
                inputs = processor(
                    text=[record["prompt"] for record in current],
                    images=images,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                for index, record in enumerate(current):
                    ids = unpad_input_ids(
                        inputs["input_ids"][index].tolist(),
                        inputs["attention_mask"][index].tolist(),
                    )
                    if ids.count(model.config.image_token_index) != expected_image_tokens:
                        raise ValueError(f"{record['id']}: incorrect expanded image-token count.")
                    if len(ids) + args.max_new_tokens > maximum_context:
                        raise ValueError(
                            f"{record['id']}: prompt plus max-new-tokens exceeds "
                            f"the {maximum_context}-token context budget; no truncation was applied."
                        )
                input_width = inputs["input_ids"].shape[1]
                inputs = {
                    key: value.to(device=device, dtype=dtype) if value.is_floating_point()
                    else value.to(device=device)
                    for key, value in inputs.items()
                }
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )
                # 只解码新生成的部分，前面的 prompt 不算模型回答。
                raw_predictions = processor.batch_decode(
                    generated[:, input_width:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for record, raw in zip(current, raw_predictions):
                    result = format_prediction(record["id"], raw)
                    parse_failures += int(result["parse_error"] is not None)
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"Predicted {min(offset + len(current), len(records))}/{len(records)}", file=sys.stderr)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {len(records)} predictions to {output}; parse failures: {parse_failures}.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        run_inference(args)
    except ModuleNotFoundError as error:
        parser.exit(2, f"Missing inference dependency: {error.name}. Install requirements-train.txt on the model host.\n")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Inference error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
