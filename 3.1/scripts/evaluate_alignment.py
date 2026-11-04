"""Evaluate the learned LLaVA image/text alignment checkpoint on original SMID splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

# 支持直接 python scripts/evaluate_alignment.py，不依赖安装本项目。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moral_llava.alignment_data import (
    alignment_manifest_hash, build_retrieval_pool, load_alignment_splits,
    manifest_fingerprints,
)
from scripts.evaluate_retrieval import average_precision, score_queries


def alignment_retrieval_metrics(
    scores: Any,
    query_labels: list[set[str]],
    caption_labels: list[set[str]],
    caption_image_ids: list[str],
    query_ids: list[str],
) -> dict[str, Any]:
    """Project moral-label R@1 plus a separately named cross-image control.

    Historical hit_at_k keys remain machine-readable compatibility aliases.
    Project R@K is the query-level success rate, not caption-level recall.
    """
    metrics = score_queries(scores, query_labels, caption_labels, caption_image_ids, query_ids)
    label_metrics = metrics["label_relevance"]
    label_metrics["definition"] = (
        "candidate and query share at least one moral label; project R@K is the fraction "
        "of image queries with at least one relevant caption in the first k results; "
        "not caption-level recall; hit_at_k is a historical compatibility alias"
    )
    for k in (1, 5, 10):
        label_metrics[f"hit_at_{k}"] = label_metrics[f"r_at_{k}"]
    label_metrics["recall_key_semantics"] = "project R@K: query-level moral-label retrieval success"

    hit_counts = {1: 0, 5: 0, 10: 0}
    average_precisions = []
    candidate_counts, relevant_counts = [], []
    for index, query_id in enumerate(query_ids):
        order = scores[index].argsort(descending=True).tolist()
        # 本图的十句（或更多/更少）全部拿掉，再看其他图片的道德标签。
        order = [position for position in order if caption_image_ids[position] != query_id]
        relevant = [bool(query_labels[index] & caption_labels[position]) for position in order]
        relevant_count = sum(relevant)
        candidate_counts.append(len(order))
        relevant_counts.append(relevant_count)
        average_precisions.append(average_precision(relevant, relevant_count))
        for k in hit_counts:
            hit_counts[k] += int(any(relevant[:k]))
    count = len(query_ids)
    metrics["cross_image_moral_relevance"] = {
        "metric_name": "cross-image R@1 (diagnostic only)",
        "definition": (
            "remove every caption belonging to the query image before ranking; a remaining "
            "caption is relevant iff it shares at least one moral label with the query"
        ),
        "hit_definition": "fraction of all image queries with at least one relevant cross-image caption among top k",
        "map_definition": "mean per-query average precision over cross-image candidates only",
        "denominator": "all image queries; queries without cross-image positives contribute zero to Hit and AP",
        "num_queries": count,
        "num_queries_with_relevant_candidates": sum(value > 0 for value in relevant_counts),
        "num_queries_without_relevant_candidates": sum(value == 0 for value in relevant_counts),
        "candidate_count_after_exclusion": {"min": min(candidate_counts), "max": max(candidate_counts)},
        "relevant_count_after_exclusion": {"min": min(relevant_counts), "max": max(relevant_counts)},
        **{f"hit_at_{k}": value / count for k, value in hit_counts.items()},
        **{f"r_at_{k}": value / count for k, value in hit_counts.items()},
        "map_i2t": sum(average_precisions) / count,
    }
    return metrics


def checkpoint_fingerprints(checkpoint: str | Path) -> dict[str, str]:
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("--checkpoint must be a saved alignment checkpoint directory")
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                hashes[str(path.relative_to(root))] = hashlib.file_digest(handle, "sha256").hexdigest()
    if not hashes:
        raise ValueError(f"Empty checkpoint directory: {root}")
    return hashes


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Saved LLaVA alignment checkpoint, not a standalone CLIP folder.")
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/SMID_moralclip"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--text-max-length", type=int, default=256, help="Reject longer captions instead of silently truncating them.")
    return parser


def evaluate_model_retrieval(
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    *,
    image_batch_size: int = 16,
    text_batch_size: int = 32,
    text_max_length: int = 256,
    progress: bool = False,
) -> dict[str, Any]:
    """Evaluate an in-memory alignment model without writing a checkpoint.

    Training code can select on result["same_image_pairing"]["map_i2t"] or
    result["label_relevance"]["map_i2t"] using validation images only.
    """
    if min(image_batch_size, text_batch_size, text_max_length) <= 0:
        raise ValueError("Batch sizes and text_max_length must be positive")
    pool = build_retrieval_pool(rows)
    import torch
    from PIL import Image, ImageOps
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_tokens = tokenizer(pool["captions"], padding=False, truncation=False)
    token_lengths = [len(ids) for ids in raw_tokens["input_ids"]]
    if max(token_lengths) > text_max_length:
        raise ValueError(
            f"A candidate caption needs {max(token_lengths)} tokens; "
            f"increase text_max_length from {text_max_length}. No captions were truncated."
        )
    image_embeddings, text_embeddings = [], []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(pool["image_paths"]), image_batch_size):
                images = []
                for path in pool["image_paths"][start:start + image_batch_size]:
                    with Image.open(path) as image:
                        images.append(ImageOps.exif_transpose(image).convert("RGB"))
                pixels = processor.image_processor(images=images, return_tensors="pt")["pixel_values"]
                encoded = model.encode_images(pixels.to(device=device, dtype=dtype))
                image_embeddings.append(encoded.float().cpu())
                if progress:
                    print(f"Images {min(start + image_batch_size, len(pool['image_paths']))}/{len(pool['image_paths'])}", file=sys.stderr)
            for start in range(0, len(pool["captions"]), text_batch_size):
                inputs = tokenizer(
                    pool["captions"][start:start + text_batch_size], padding=True,
                    truncation=False, return_tensors="pt",
                )
                encoded = model.encode_texts(
                    inputs["input_ids"].to(device), inputs["attention_mask"].to(device),
                )
                text_embeddings.append(encoded.float().cpu())
                if progress:
                    print(f"Captions {min(start + text_batch_size, len(pool['captions']))}/{len(pool['captions'])}", file=sys.stderr)
    finally:
        model.train(was_training)
    image_embeddings = torch.cat(image_embeddings)
    text_embeddings = torch.cat(text_embeddings)
    if (
        image_embeddings.ndim != 2 or text_embeddings.ndim != 2
        or image_embeddings.shape[1] != text_embeddings.shape[1]
        or image_embeddings.shape[0] != len(pool["query_ids"])
        or text_embeddings.shape[0] != len(pool["captions"])
        or not torch.isfinite(image_embeddings).all()
        or not torch.isfinite(text_embeddings).all()
    ):
        raise ValueError("Checkpoint returned invalid image/text embeddings")
    image_norms = image_embeddings.norm(dim=-1)
    text_norms = text_embeddings.norm(dim=-1)
    if (image_norms < 1e-12).any() or (text_norms < 1e-12).any():
        raise ValueError("Checkpoint returned a zero embedding")
    image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    scores = image_embeddings @ text_embeddings.T
    metrics = alignment_retrieval_metrics(
        scores, pool["query_labels"], pool["caption_labels"],
        pool["caption_image_ids"], pool["query_ids"],
    )
    metrics["embedding_dimension"] = int(image_embeddings.shape[1])
    metrics["text_token_lengths"] = {"min": min(token_lengths), "max": max(token_lengths), "truncated": 0}
    return metrics


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.image_batch_size, args.text_batch_size, args.text_max_length) <= 0:
        raise ValueError("Batch sizes and text-max-length must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_hashes = checkpoint_fingerprints(checkpoint)
    splits = load_alignment_splits(args.image_root)
    pool = build_retrieval_pool(splits[args.split])
    import torch
    from moral_llava.alignment import load_alignment_checkpoint

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available")
    if args.precision == "auto":
        dtype = (
            torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16 if device == "cuda" else torch.float32
        )
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    if dtype == torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")
    # 必须重载实际训练的图文两路及投影头，不能拿原MoralCLIP的分数顶替。
    model, processor, metadata = load_alignment_checkpoint(
        checkpoint, device=device, dtype=dtype, local_files_only=True, trainable=False,
    )
    metrics = evaluate_model_retrieval(
        model, processor, splits[args.split], image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size, text_max_length=args.text_max_length,
        progress=True,
    )
    result = {
        "status": "PASS",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "LLaVA learned image-text alignment checkpoint",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hashes,
        "checkpoint_metadata": metadata,
        "image_root": str(args.image_root.expanduser().resolve()),
        "original_split": args.split,
        "manifest_sha256": manifest_fingerprints(args.image_root),
        "combined_manifest_sha256": alignment_manifest_hash(args.image_root),
        "metric_direction": "image_to_text",
        "metric_name": "R@1",
        "protocol": "moral-label-r1-v1",
        "r_at_1": metrics["label_relevance"]["r_at_1"],
        "score": "cosine similarity of normalized learned LLaVA image/text embeddings",
        "candidate_pool": f"all original captions of the {args.split} image split",
        "recall_definition": "fraction of image queries with at least one relevant caption among top k",
        "query_ids": pool["query_ids"],
        "candidate_ids": pool["caption_ids"],
        "candidate_image_ids": pool["caption_image_ids"],
        "num_queries": len(pool["query_ids"]),
        "num_candidates": len(pool["captions"]),
        "caption_count_distribution": pool["caption_count_distribution"],
        "embedding_dimension": metrics["embedding_dimension"],
        "precision": str(dtype),
        "text_token_lengths": metrics["text_token_lengths"],
        "metrics": metrics,
        "scope": "Alignment retrieval only; this metric does not measure VQA or five-foundation generation accuracy.",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(args)
    except ModuleNotFoundError as error:
        parser.exit(2, f"Missing evaluation dependency: {error.name}\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"Evaluation error: {error}\n")
    print(json.dumps({"checkpoint": result["checkpoint"], "metrics": result["metrics"]}, ensure_ascii=False, indent=2))
    print(f"Saved evaluation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
