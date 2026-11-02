"""Rank candidate captions for an image with the learned alignment checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--captions-file", type=Path, help="UTF-8 text, one candidate description per line")
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/SMID_moralclip"))
    parser.add_argument("--candidate-split", choices=("validation", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.top_k, args.text_batch_size, args.text_max_length) < 1:
        parser.error("top-k, text-batch-size and text-max-length must be positive")
    import torch
    from PIL import Image, ImageOps
    from moral_llava.alignment import load_alignment_checkpoint

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA GPU unavailable")
    if args.precision == "bf16" and (args.device != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("bf16 needs a supported CUDA GPU; use --precision fp32 for CPU")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    candidate_ids = None
    if args.captions_file:
        captions = [line.strip() for line in args.captions_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        source = str(args.captions_file.resolve())
    else:
        from moral_llava.alignment_data import build_retrieval_pool, load_alignment_splits

        pool = build_retrieval_pool(load_alignment_splits(args.image_root)[args.candidate_split])
        captions = pool["captions"]
        candidate_ids = pool["caption_ids"]
        source = f"all captions of original {args.candidate_split} split"
    if not captions:
        parser.error("Candidate descriptions are empty")
    model, processor, record = load_alignment_checkpoint(
        args.checkpoint, device=args.device, dtype=dtype, local_files_only=True, trainable=False,
    )
    model.eval()
    tokenizer = processor.tokenizer
    with Image.open(args.image) as image:
        pixels = processor.image_processor(images=ImageOps.exif_transpose(image).convert("RGB"), return_tensors="pt")["pixel_values"]
    scores = []
    with torch.inference_mode():
        image_vector = model.encode_images(pixels.to(args.device, dtype=dtype))
        for start in range(0, len(captions), args.text_batch_size):
            inputs = tokenizer(captions[start:start + args.text_batch_size], padding=True,
                               truncation=False, return_tensors="pt")
            if inputs["input_ids"].shape[1] > args.text_max_length:
                parser.error("Candidate exceeds text-max-length; increase it, no text has been truncated")
            text_vectors = model.encode_texts(inputs["input_ids"].to(args.device), inputs["attention_mask"].to(args.device))
            scores.append((image_vector @ text_vectors.T).squeeze(0).float().cpu())
    scores = torch.cat(scores)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite retrieval scores")
    order = scores.argsort(descending=True)[:min(args.top_k, len(captions))].tolist()
    result = {
        "image": str(args.image.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "training_run_kind": record.get("metadata", {}).get("run_config", {}).get("run_kind"),
        "candidate_source": source, "candidate_count": len(captions),
        "score_definition": "cosine similarity, not a calibrated probability",
        "results": [{"rank": rank, "caption": captions[index], "cosine": float(scores[index]),
                     **({"candidate_id": candidate_ids[index]} if candidate_ids else {})}
                    for rank, index in enumerate(order, start=1)],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
