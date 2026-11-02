"""Evaluate SMID image-to-text retrieval with a CLIP-style dual encoder.

论文口径按共享至少一个 moral label 计相关；另外输出同图配对口径，免得
客户把两个 R@1 混在一起。这个脚本评估的是指定的双编码器，不是 LLaVA
生成式分类 adapter 的输出。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("manifest 必须是非空数组")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("manifest 中有无效样本")
        captions = row.get("captions")
        labels = row.get("moral_labels")
        if not isinstance(captions, list) or not captions or not all(isinstance(c, str) and c for c in captions):
            raise ValueError(f"{row['id']} 的 captions 不完整")
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError(f"{row['id']} 的 moral_labels 不完整")
    return rows


def normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def average_precision(ranked_relevant: Sequence[bool], relevant_count: int) -> float:
    if relevant_count == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, relevant in enumerate(ranked_relevant, start=1):
        if relevant:
            hits += 1
            total += hits / rank
    return total / relevant_count


def score_queries(
    scores: torch.Tensor,
    query_labels: list[set[str]],
    caption_labels: list[set[str]],
    caption_image_ids: list[str],
    query_ids: list[str],
) -> dict[str, Any]:
    topk = {1: [], 5: [], 10: []}
    paired_topk = {1: [], 5: [], 10: []}
    aps: list[float] = []
    paired_aps: list[float] = []
    top1_label: list[bool] = []
    top1_paired: list[bool] = []
    for index, query_id in enumerate(query_ids):
        order = scores[index].argsort(descending=True).tolist()
        relevant = [bool(query_labels[index] & caption_labels[position]) for position in order]
        paired = [caption_image_ids[position] == query_id for position in order]
        for k in topk:
            topk[k].append(float(any(relevant[:k])))
            paired_topk[k].append(float(any(paired[:k])))
        aps.append(average_precision(relevant, sum(relevant)))
        paired_aps.append(average_precision(paired, sum(paired)))
        top1_label.append(relevant[0])
        top1_paired.append(paired[0])
    return {
        "num_queries": len(query_ids),
        "num_candidates": len(caption_labels),
        "label_relevance": {
            "definition": "candidate and query share at least one moral label",
            "r_at_1": sum(top1_label) / len(top1_label),
            "r_at_5": sum(topk[5]) / len(topk[5]),
            "r_at_10": sum(topk[10]) / len(topk[10]),
            "map_i2t": sum(aps) / len(aps),
        },
        "same_image_pairing": {
            "definition": "candidate caption belongs to the query image",
            "r_at_1": sum(top1_paired) / len(top1_paired),
            "r_at_5": sum(paired_topk[5]) / len(paired_topk[5]),
            "r_at_10": sum(paired_topk[10]) / len(paired_topk[10]),
            "map_i2t": sum(paired_aps) / len(paired_aps),
        },
    }


def main() -> int:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", default="anaaa2/moralclip-base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前没有可用 GPU")
    rows = load_manifest(args.manifest)
    query_ids = [row["id"] for row in rows]
    query_labels = [set(row["moral_labels"]) for row in rows]
    image_paths = [args.image_root / row["image_path"] for row in rows]
    if any(not path.is_file() for path in image_paths):
        missing = [str(path) for path in image_paths if not path.is_file()][:5]
        raise FileNotFoundError(f"缺少测试图片：{missing}")
    captions: list[str] = []
    caption_labels: list[set[str]] = []
    caption_image_ids: list[str] = []
    for row in rows:
        for caption in row["captions"]:
            captions.append(caption)
            caption_labels.append(set(row["moral_labels"]))
            caption_image_ids.append(row["id"])
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=False)
    model = CLIPModel.from_pretrained(args.model, local_files_only=False).to(args.device).eval()
    image_embeddings = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), args.image_batch_size):
            images = [Image.open(path).convert("RGB") for path in image_paths[start:start + args.image_batch_size]]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            image_embeddings.append(normalize(model.get_image_features(**inputs)).float().cpu())
        text_embeddings = []
        for start in range(0, len(captions), args.text_batch_size):
            inputs = processor(text=captions[start:start + args.text_batch_size], return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            text_embeddings.append(normalize(model.get_text_features(**inputs)).float().cpu())
    scores = torch.cat(image_embeddings) @ torch.cat(text_embeddings).T
    result = {
        "status": "PASS",
        "model": args.model,
        "metric_direction": "image_to_text",
        "candidate_pool": "test captions only",
        "caption_count_per_image": 10,
        "score": "cosine similarity of normalized dual-encoder embeddings",
        "metrics": score_queries(scores, query_labels, caption_labels, caption_image_ids, query_ids),
        "warning": "This is a dual-encoder reference evaluation; the current LLaVA LoRA classifier has no shared image-text embedding space.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
