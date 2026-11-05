"""Train LLaVA-based image/text retrieval with an optional moral constraint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Sequence

from moral_llava.alignment_data import AlignmentDataset, load_alignment_splits, manifest_fingerprints
from moral_llava.alignment_losses import moral_alignment_loss


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class AlignmentCollator:
    def __init__(self, processor: Any, text_max_length: int) -> None:
        self.processor = processor
        self.text_max_length = text_max_length

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from PIL import Image, ImageOps

        images = []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        tokens = self.processor.tokenizer(
            [row["caption"] for row in rows], padding=True, truncation=False, return_tensors="pt",
        )
        if tokens["input_ids"].shape[1] > self.text_max_length:
            raise ValueError("Caption exceeds text-max-length; increase it rather than dropping text")
        return {
            "pixel_values": self.processor.image_processor(images=images, return_tensors="pt")["pixel_values"],
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "label_vectors": torch.tensor([row["label_vector"] for row in rows]),
            "image_ids": [row["id"] for row in rows],
        }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/SMID_moralclip"))
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory, separate from old SFT weights")
    parser.add_argument("--base-model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--revision", default="b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
    parser.add_argument("--vision-model", default=None, help="Optional full CLIP/CLIPVision checkpoint, e.g. local MoralCLIP B/16")
    parser.add_argument("--vision-revision", default=None)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--freeze-projector", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64, help="Actual contrastive matrix size; gradient accumulation does not enlarge it")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.)
    parser.add_argument("--max-grad-norm", type=float, default=1.)
    parser.add_argument("--moral-weight", type=float, default=.4, help="Use 0 for the matched CLIP-loss-only ablation")
    parser.add_argument("--moral-similarity", choices=("paper_scaled", "cosine"), default="paper_scaled")
    parser.add_argument("--temperature", type=float, default=.07)
    parser.add_argument("--train-temperature", action="store_true", help="Optional deviation from fixed paper temperature")
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--validation-text-batch-size", type=int, default=32)
    parser.add_argument("--validation-image-batch-size", type=int, default=32)
    parser.add_argument("--selection-metric", choices=("moral_r1", "moral_map", "paired_r1"), default="moral_r1",
                        help="Default: project moral-label R@1 on validation; paired_r1 is a separate diagnostic")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Default uses existing local model cache")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="0 trains all epochs; positive values are marked partial_training")
    parser.add_argument("--smoke-test", action="store_true", help="Only two optimizer steps and 8 validation images; not a benchmark")
    return parser


def selection_value(metrics: dict[str, Any], name: str) -> float:
    if name == "moral_r1":
        return metrics["label_relevance"]["r_at_1"]
    if name == "moral_map":
        return metrics["label_relevance"]["map_i2t"]
    if name == "paired_r1":
        return metrics["same_image_pairing"]["r_at_1"]
    raise ValueError(f"Unknown selection metric: {name}")


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from moral_llava.alignment import load_alignment_model, save_alignment_checkpoint
    from scripts.evaluate_alignment import evaluate_model_retrieval

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This entry point is single-process; it does not gather negatives across GPUs")
    if min(args.epochs, args.gradient_accumulation, args.text_max_length, args.log_every,
           args.validation_text_batch_size, args.validation_image_batch_size) < 1:
        raise ValueError("Epochs, lengths, logging interval and accumulation must be positive")
    if args.batch_size < 2 or args.num_workers < 0 or args.max_steps < 0:
        raise ValueError("batch-size must be >=2; num-workers/max-steps must be nonnegative")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or not math.isfinite(args.moral_weight) or args.moral_weight < 0
            or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0
            or not 0 <= args.warmup_ratio < 1):
        raise ValueError("Invalid optimizer/loss configuration")
    if args.vision_model and args.freeze_projector:
        raise ValueError("A replaced vision tower needs its new projector trained")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("No CUDA GPU is available")
    if args.precision == "bf16" and (args.device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 training requires a supported CUDA GPU; use --precision fp32 for CPU")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    # 训练阶段只打开 train 和 validation，测试集留到配置锁定后再评。
    data_splits = ("train", "validation")
    splits = load_alignment_splits(args.image_root, included_splits=data_splits)
    train_rows, validation_rows = splits["train"], splits["validation"]
    run_kind = "smoke_test" if args.smoke_test else "partial_training" if args.max_steps else "training"
    if args.smoke_test:
        train_rows = train_rows[:max(2 * args.batch_size, 8)]
        validation_rows = validation_rows[:8]
        args.epochs, args.max_steps = 1, 2
    if len(train_rows) < args.batch_size:
        raise ValueError("Training split is smaller than the actual contrastive batch")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "run_kind": run_kind,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "objective": "symmetric paired CLIP CE + lambda * off-diagonal moral Jaccard MSE",
        "architecture": "image: vision->mm_projector->mean->head; text: Vicuna+LoRA->masked_mean->head",
        "actual_contrastive_batch": args.batch_size,
        "optimizer_effective_batch": args.batch_size * args.gradient_accumulation,
        "manifest_sha256": manifest_fingerprints(args.image_root, included_splits=data_splits),
        "data_access_splits": list(data_splits),
        "original_split_sizes": {key: len(rows) for key, rows in splits.items()},
        "used_train_images": len(train_rows), "used_validation_images": len(validation_rows),
        "test_used_for_selection": False,
        "drop_last": True,
        "checkpoint_contains_optimizer_state": False,
    })
    if args.vision_model:
        source_path = Path(str(Path(args.vision_model).expanduser()) + ".source.json")
        if source_path.is_file():
            config["vision_initialization_source"] = json.loads(source_path.read_text(encoding="utf-8"))
    write_json(output / "run_config.json", config)
    print(json.dumps({"output": str(output), "run_kind": run_kind, "actual_contrastive_batch": args.batch_size}, ensure_ascii=False), flush=True)
    model, processor = load_alignment_model(
        args.base_model, revision=args.revision, vision_model_id=args.vision_model,
        vision_revision=args.vision_revision, projection_dim=args.projection_dim,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        train_projector=not args.freeze_projector, temperature=args.temperature,
        train_logit_scale=args.train_temperature, dtype=dtype, device=args.device,
        local_files_only=not args.allow_download, trainable=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_counts = {}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        captions = [caption for row in rows for caption in row["captions"]]
        lengths = [len(ids) for ids in tokenizer(captions, truncation=False)["input_ids"]]
        token_counts[split] = {"min": min(lengths), "max": max(lengths), "truncated": 0}
        if max(lengths) > args.text_max_length:
            raise ValueError(f"{split} caption needs {max(lengths)} tokens; increase --text-max-length")
    config["text_token_lengths"] = token_counts
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["trainable_parameter_names"] = model.trainable_parameter_names
    write_json(output / "run_config.json", config)
    dataset = AlignmentDataset(train_rows, seed=args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=AlignmentCollator(processor, args.text_max_length),
        num_workers=args.num_workers, persistent_workers=False,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=args.device == "cuda",
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(loader) / args.gradient_accumulation)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1., max(0., (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return .5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    global_step, best_value, best_checkpoint = 0, -math.inf, None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        sums = {name: 0. for name in ("loss", "clip_loss", "moral_loss", "weighted_moral_loss")}
        batches = 0
        for batch_index, batch in enumerate(loader):
            inputs = {name: batch[name].to(args.device, non_blocking=True)
                      for name in ("pixel_values", "input_ids", "attention_mask")}
            encoded = model(**inputs)
            losses = moral_alignment_loss(
                encoded["logits_per_image"], batch["label_vectors"],
                moral_weight=args.moral_weight, moral_similarity=args.moral_similarity,
                image_embeddings=encoded["image_embeddings"], text_embeddings=encoded["text_embeddings"],
                image_ids=batch["image_ids"],
            )
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("Nonfinite training loss")
            # 累积只影响优化器更新，不会把几批的负样本自动拼到一起。
            window_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(loader) - window_start)
            (losses["loss"] / window_size).backward()
            batches += 1
            for key in sums:
                sums[key] += float(losses[key].detach())
            update = (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader)
            if not update:
                continue
            if global_step == 0:
                gradient_norms = {"image_projection": 0., "text_projection": 0., "language_lora": 0., "mm_projector": 0.}
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    group = ("image_projection" if name.startswith("image_projection.") else
                             "text_projection" if name.startswith("text_projection.") else
                             "language_lora" if "lora_" in name else
                             "mm_projector" if "multi_modal_projector" in name else None)
                    if group:
                        gradient_norms[group] += parameter.grad.detach().float().square().sum().item()
                gradient_norms = {key: math.sqrt(value) for key, value in gradient_norms.items()}
                if any(not math.isfinite(gradient_norms[key]) or gradient_norms[key] <= 0
                       for key in ("image_projection", "text_projection", "language_lora")):
                    raise RuntimeError(f"A retrieval branch is not receiving finite gradients: {gradient_norms}")
                write_json(output / "first_step_gradients.json", gradient_norms)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            event = {
                "epoch": epoch + 1, "optimizer_step": global_step,
                "losses": {key: float(losses[key].detach()) for key in sums},
                "learning_rate": learning_rate, "gradient_norm_before_clip": float(grad_norm),
                "logit_scale": float(encoded["logit_scale"].detach()),
                "actual_contrastive_batch": len(batch["image_ids"]),
            }
            with (output / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            if global_step == 1 or global_step % args.log_every == 0:
                print(json.dumps(event), flush=True)
            if global_step >= total_steps:
                break
        metrics = evaluate_model_retrieval(
            model, processor, validation_rows,
            image_batch_size=args.validation_image_batch_size,
            text_batch_size=args.validation_text_batch_size, text_max_length=args.text_max_length,
        )
        selected = selection_value(metrics, args.selection_metric)
        validation = {
            "epoch": epoch + 1, "optimizer_step": global_step, "run_kind": run_kind,
            "split": "validation", "metrics": metrics,
            "mean_train_losses": {key: value / batches for key, value in sums.items()},
            "selection_metric": args.selection_metric, "selection_value": selected,
        }
        write_json(output / f"validation_epoch_{epoch + 1:03d}.json", validation)
        checkpoint = output / f"epoch-{epoch + 1:03d}"
        save_alignment_checkpoint(model, processor, checkpoint, metadata={
            "run_config": config, "validation": validation, "optimizer_step": global_step,
        })
        if selected > best_value:
            best_value, best_checkpoint = selected, checkpoint
            write_json(output / "best_checkpoint.json", {
                "checkpoint": str(checkpoint), "selection_metric": args.selection_metric,
                "selection_value": selected, "split": "validation", "run_kind": run_kind,
            })
        print(json.dumps({"epoch": epoch + 1, "validation": metrics, "checkpoint": str(checkpoint)}, ensure_ascii=False), flush=True)
        if global_step >= total_steps:
            break
    summary = {
        "status": "PASS", "run_kind": run_kind, "optimizer_steps": global_step,
        "best_checkpoint": str(best_checkpoint), "selection_metric": args.selection_metric,
        "best_validation_value": best_value, "test_evaluated": False,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.device == "cuda":
        summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    write_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    print(json.dumps(train(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
