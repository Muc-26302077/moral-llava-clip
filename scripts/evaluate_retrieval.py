"""Evaluate bidirectional project R@1 with shared moral-label relevance.

R@1 is the project name for the former moral-label Hit@1. The candidate pool
includes the query image's captions. Paired retrieval and cross-image retrieval
are separately named diagnostics; neither replaces the project R@1 baseline.
This entry evaluates the explicitly supplied standalone MoralCLIP checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MORAL_LABELS = {
    "Care", "Harm", "Fairness", "Cheating", "Loyalty", "Betrayal",
    "Respect", "Subversion", "Sanctity", "Degradation", "non-moral",
}

def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("manifest 必须是非空数组")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("manifest 中有无效样本")
        captions = row.get("captions")
        labels = row.get("moral_labels")
        if not isinstance(captions, list) or not captions or not all(isinstance(c, str) and c.strip() for c in captions):
            raise ValueError(f"{row['id']} 的 captions 不完整")
        if (not isinstance(labels, list) or not labels
                or any(not isinstance(label, str) or label not in MORAL_LABELS for label in labels)
                or ("non-moral" in labels and len(labels) != 1)):
            raise ValueError(f"{row['id']} 的 moral_labels 不完整")
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError("manifest 包含重复图像 ID")
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
    *,
    direction: str = "image_to_text",
) -> dict[str, Any]:
    if direction not in ("image_to_text", "text_to_image"):
        raise ValueError("Unknown retrieval direction")
    if not query_ids or not caption_image_ids:
        raise ValueError("Queries and candidate captions must be nonempty")
    if len(query_ids) != len(query_labels) or len(caption_image_ids) != len(caption_labels):
        raise ValueError("Query/candidate metadata counts disagree")
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
        "primary_metric": "R@1",
        "metric_direction": direction,
        "r_at_1": sum(top1_label) / len(top1_label),
        "correct_queries": sum(top1_label),
        "num_queries": len(query_ids),
        "num_candidates": len(caption_labels),
        "label_relevance": {
            "metric_name": "R@1",
            "definition": "candidate and query share at least one moral label",
            "r_at_1": sum(top1_label) / len(top1_label),
            "r_at_5": sum(topk[5]) / len(topk[5]),
            "r_at_10": sum(topk[10]) / len(topk[10]),
            ("map_i2t" if direction == "image_to_text" else "map_t2i"): sum(aps) / len(aps),
        },
        "same_image_pairing": {
            "metric_name": "paired R@1 (diagnostic only)",
            "definition": ("candidate caption belongs to the query image" if direction == "image_to_text"
                           else "candidate image is the annotated source image of the query caption"),
            "r_at_1": sum(top1_paired) / len(top1_paired),
            "r_at_5": sum(paired_topk[5]) / len(paired_topk[5]),
            "r_at_10": sum(paired_topk[10]) / len(paired_topk[10]),
            ("map_i2t" if direction == "image_to_text" else "map_t2i"): sum(paired_aps) / len(paired_aps),
        },
    }


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Explicit original MoralCLIP HF directory; no implicit checkpoint")
    parser.add_argument("--checkpoint-file", type=Path, help="Optional complete HF CLIP state dict")
    parser.add_argument("--processor", help="Matching processor directory; defaults to --model")
    parser.add_argument("--expected-model-sha256", help="Require this SHA-256 for a single weight file")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly allow HF downloads; local-only by default")
    parser.add_argument("--revision", help="Pinned HF revision when downloading the explicit model")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/SMID_moralclip")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--output", type=Path, required=True, help="New metrics JSON; also saves predictions, scores and report")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    import torch
    import transformers
    from PIL import Image
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor

    if min(args.image_batch_size, args.text_batch_size) < 1:
        parser.error('Batch sizes must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('请求了 CUDA，但当前没有可用 GPU')
    output = args.output.expanduser().resolve()
    predictions_path = output.with_suffix('.predictions.json')
    reverse_predictions_path = output.with_suffix('.text_to_image.predictions.json')
    scores_path = output.with_suffix('.scores.npy')
    report_path = output.with_suffix('.report.md')
    paths = (output, predictions_path, reverse_predictions_path, scores_path, report_path)
    if any(path.exists() for path in paths):
        raise FileExistsError('Output already exists; use a new --output to preserve prior evidence')
    image_root = args.image_root.expanduser().resolve()
    manifest = (args.manifest or (image_root / ('test/smid_manifest.json' if args.split == 'test'
                                               else 'smid_manifest.json'))).resolve()
    rows = [r for r in load_manifest(manifest) if r.get('split') == args.split]
    if not rows:
        raise ValueError(f'No {args.split} rows in manifest')
    query_ids = [r['id'] for r in rows]
    query_labels = [set(r['moral_labels']) for r in rows]
    image_paths = [(image_root / r['image_path']).resolve() for r in rows]
    for path in image_paths:
        if not path.is_relative_to(image_root):
            raise ValueError(f'Image outside --image-root: {path}')
        if not path.is_file():
            raise FileNotFoundError(f'Missing image: {path}')
    captions = [caption for r in rows for caption in r['captions']]
    caption_labels = [set(r['moral_labels']) for r in rows for _ in r['captions']]
    caption_image_ids = [r['id'] for r in rows for _ in r['captions']]
    caption_indices = [i for r in rows for i in range(len(r['captions']))]
    local = not args.allow_download
    model_source = args.model
    if local and not Path(model_source).expanduser().is_dir():
        raise FileNotFoundError('Provide a local --model directory, or explicitly use --allow-download')
    if Path(model_source).expanduser().is_dir():
        model_source = str(Path(model_source).expanduser().resolve())
    if args.checkpoint_file:
        weight_paths = [args.checkpoint_file.expanduser().resolve()]
    elif Path(model_source).is_dir():
        model_dir = Path(model_source)
        safe = sorted(p for p in model_dir.glob('*.safetensors') if not p.name.startswith('._'))
        weight_paths = safe or sorted(model_dir.glob('pytorch_model*.bin'))
    else:
        # Pin and record the resolved HF revision rather than an unversioned download.
        from huggingface_hub import snapshot_download
        model_source = snapshot_download(repo_id=model_source, revision=args.revision)
        model_dir = Path(model_source)
        weight_paths = sorted(model_dir.glob('*.safetensors')) or sorted(model_dir.glob('pytorch_model*.bin'))
    if not weight_paths:
        raise FileNotFoundError('No complete model weights found in the explicit checkpoint')
    hashes = {p.name: sha256(p) for p in weight_paths}
    if args.expected_model_sha256 and (len(hashes) != 1 or next(iter(hashes.values())) != args.expected_model_sha256):
        raise ValueError('Model SHA-256 differs from the required original checkpoint')
    print(f'Loading original checkpoint: {model_source}; {len(rows)} images / {len(captions)} captions', flush=True)
    if args.checkpoint_file:
        from safetensors.torch import load_file
        model = CLIPModel(CLIPConfig.from_pretrained(model_source, local_files_only=local))
        path = weight_paths[0]
        state = load_file(str(path)) if path.suffix == '.safetensors' else torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(state, dict):
            for key in ('state_dict', 'model_state_dict'):
                if key in state:
                    state = state[key]
                    break
        if not isinstance(state, dict) or not state:
            raise ValueError('Expected a complete HF CLIP state dict')
        if all(k.startswith('module.') for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        loading_info = {}
    else:
        model, loading_info = CLIPModel.from_pretrained(model_source, local_files_only=local, output_loading_info=True)
        errors = {k: loading_info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
                  if loading_info.get(k)}
        if errors:
            raise RuntimeError(f'Incomplete checkpoint loading: {errors}')
    processor_source = args.processor or model_source
    processor = CLIPProcessor.from_pretrained(processor_source, local_files_only=local, use_fast=False)
    model = model.to(args.device).eval()
    max_length = model.config.text_config.max_position_embeddings
    token_ids = processor.tokenizer(captions, truncation=False, padding=False)['input_ids']
    long_count = sum(len(ids) > max_length for ids in token_ids)
    image_embeddings, text_embeddings = [], []
    with torch.inference_mode():
        for start in range(0, len(image_paths), args.image_batch_size):
            images = []
            for path in image_paths[start:start + args.image_batch_size]:
                with Image.open(path) as im:
                    images.append(im.convert('RGB'))
            inputs = processor(images=images, return_tensors='pt').to(args.device)
            image_embeddings.append(normalize(model.get_image_features(**inputs).float()).cpu())
            for im in images:
                im.close()
            print(f'Images {min(start + args.image_batch_size, len(rows))}/{len(rows)}', flush=True)
        for start in range(0, len(captions), args.text_batch_size):
            inputs = processor(text=captions[start:start + args.text_batch_size], return_tensors='pt',
                               padding=True, truncation=True, max_length=max_length).to(args.device)
            text_embeddings.append(normalize(model.get_text_features(**inputs).float()).cpu())
    im, tx = torch.cat(image_embeddings), torch.cat(text_embeddings)
    if (not torch.isfinite(im).all() or not torch.isfinite(tx).all()
            or (im.norm(dim=-1) < 1e-12).any() or (tx.norm(dim=-1) < 1e-12).any()):
        raise ValueError('Checkpoint produced invalid/zero embeddings')
    scores = im @ tx.T
    metrics = score_queries(scores, query_labels, caption_labels, caption_image_ids, query_ids)
    reverse_metrics = score_queries(scores.T, caption_labels, query_labels, query_ids, caption_image_ids,
                                    direction='text_to_image')
    records = []
    cross_hits = 0
    for i, query_id in enumerate(query_ids):
        order = scores[i].argsort(descending=True).tolist()
        j = order[0]
        cross_order = [p for p in order if caption_image_ids[p] != query_id]
        cross_j = cross_order[0] if cross_order else None
        cross_hit = cross_j is not None and bool(query_labels[i] & caption_labels[cross_j])
        cross_hits += int(cross_hit)
        records.append({
            'query_id': query_id, 'query_labels': sorted(query_labels[i]),
            'retrieved_image_id': caption_image_ids[j], 'caption_index': caption_indices[j],
            'retrieved_caption': captions[j], 'retrieved_labels': sorted(caption_labels[j]),
            'shared_labels': sorted(query_labels[i] & caption_labels[j]),
            'r_at_1_correct': bool(query_labels[i] & caption_labels[j]),
            'paired_r_at_1_correct': caption_image_ids[j] == query_id,
            'cosine_similarity': float(scores[i, j]),
            'top1_tie_count': int((scores[i] == scores[i, j]).sum()),
            'cross_image': None if cross_j is None else {
                'retrieved_image_id': caption_image_ids[cross_j], 'caption_index': caption_indices[cross_j],
                'retrieved_labels': sorted(caption_labels[cross_j]), 'r_at_1_correct': cross_hit,
                'candidate_count': len(cross_order), 'cosine_similarity': float(scores[i, cross_j]),
            },
        })
    # Independent recount from saved predictions, using the named project relevance rule.
    hits = sum(bool(set(r['query_labels']) & set(r['retrieved_labels'])) for r in records)
    if hits / len(records) != metrics['r_at_1']:
        raise RuntimeError('Independent R@1 recalculation disagrees')
    reverse_records = []
    for j, caption in enumerate(captions):
        i = int(scores[:, j].argsort(descending=True)[0])
        reverse_records.append({
            'query_id': f'{caption_image_ids[j]}::caption::{caption_indices[j]}',
            'query_image_id': caption_image_ids[j], 'caption_index': caption_indices[j],
            'query_text': caption, 'query_labels': sorted(caption_labels[j]),
            'retrieved_image_id': query_ids[i], 'retrieved_labels': sorted(query_labels[i]),
            'shared_labels': sorted(caption_labels[j] & query_labels[i]),
            'r_at_1_correct': bool(caption_labels[j] & query_labels[i]),
            'paired_r_at_1_correct': caption_image_ids[j] == query_ids[i],
            'cosine_similarity': float(scores[i, j]),
            'top1_tie_count': int((scores[:, j] == scores[i, j]).sum()),
        })
    reverse_hits = sum(bool(set(r['query_labels']) & set(r['retrieved_labels'])) for r in reverse_records)
    if reverse_hits / len(reverse_records) != reverse_metrics['r_at_1']:
        raise RuntimeError('Independent text-to-image R@1 recalculation disagrees')
    groups = {}
    for name, subset in [('all', records), ('moral_queries', [r for r in records if r['query_labels'] != ['non-moral']]),
                         ('non_moral_queries', [r for r in records if r['query_labels'] == ['non-moral']])]:
        count = len(subset)
        groups[name] = {'num_queries': count, 'correct_queries': sum(r['r_at_1_correct'] for r in subset),
                        'r_at_1': sum(r['r_at_1_correct'] for r in subset) / count if count else None}
    result = {
        'status': 'PASS', 'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'metric_name': 'R@1', 'protocol': 'moral-label-r1-v1',
        'format_version': 2,
        'definition': 'for each direction, fraction of queries whose top-1 result shares at least one moral label',
        'former_name': 'moral-label Hit@1', 'r_at_1': metrics['r_at_1'],
        'correct_queries': hits, 'num_queries': len(rows), 'num_candidates': len(captions),
        'model': model_source, 'processor': processor_source, 'model_weight_sha256': hashes,
        'requested_model': args.model, 'requested_revision': args.revision,
        'model_config': model.config.to_dict(),
        'model_and_processor_files_sha256': {p.name: sha256(p) for p in Path(model_source).glob('*')
                                             if p.is_file() and not p.name.startswith('.') and p.suffix in ('.json', '.txt')},
        'loading_errors': {k: loading_info.get(k, []) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')},
        'manifest_sha256': sha256(manifest), 'evaluation_script_sha256': sha256(Path(__file__)),
        'image_sha256': {r['id']: sha256(p) for r, p in zip(rows, image_paths)},
        'metric_direction': 'bidirectional', 'original_split': args.split,
        'directions': {'image_to_text': metrics, 'text_to_image': reverse_metrics},
        'legacy_flat_metrics_direction': 'image_to_text',
        'candidate_pool': 'image_to_text: all split captions; text_to_image: all split images; paired candidates retained',
        'non_moral_policy': 'included as an ordinary relevance label',
        'caption_count_distribution': sorted({len(r['captions']) for r in rows}),
        'text_max_length': max_length, 'truncated_captions': long_count,
        'score': 'cosine similarity of normalized dual-encoder embeddings',
        'tie_policy': 'torch descending argsort; tied first-place queries separately counted',
        'queries_with_top1_ties': sum(r['top1_tie_count'] > 1 for r in records),
        'text_to_image_queries_with_top1_ties': sum(r['top1_tie_count'] > 1 for r in reverse_records),
        'metrics': metrics, 'groups': groups,
        'cross_image_diagnostic': {'metric_name': 'cross-image R@1', 'r_at_1': cross_hits / len(rows),
                                   'correct_queries': cross_hits, 'num_queries': len(rows),
                                   'candidate_pool': 'remove all own-image captions separately for each query'},
        'independent_recount_verified': True,
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
        'scope': 'standalone MoralCLIP retrieval; not TICR or polarity classification accuracy',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.save(scores_path, scores.numpy())
    write_json(predictions_path, records)
    write_json(reverse_predictions_path, reverse_records)
    write_json(output, result)
    report = f'''# MoralCLIP 项目 R@1 基准实测

本项目 R@1 为原道德标签 Hit@1 的正式名称：在图到文和文到图检索中，首位结果与查询场景共享至少一个道德标签即命中。两个方向分别报告，不合并分母。保留原配候选及 non-moral 标签。

| 测试项 | 命中数/查询数 | 结果 |
| --- | ---: | ---: |
| 图到文 R@1（正式指标，{len(captions)} 条候选文本） | {hits}/{len(rows)} | {result['r_at_1']:.2%} |
| 文到图 R@1（正式指标，{len(rows)} 张候选图像） | {reverse_hits}/{len(captions)} | {reverse_metrics['r_at_1']:.2%} |
| 图到文同图配对 R@1（诊断） | {sum(r['paired_r_at_1_correct'] for r in records)}/{len(rows)} | {metrics['same_image_pairing']['r_at_1']:.2%} |
| 文到图原配 R@1（诊断） | {sum(r['paired_r_at_1_correct'] for r in reverse_records)}/{len(captions)} | {reverse_metrics['same_image_pairing']['r_at_1']:.2%} |
| 跨场景 R@1（排除本图描述，诊断） | {cross_hits}/{len(rows)} | {cross_hits / len(rows):.2%} |

- 模型：`{model_source}`，完整权重 SHA-256：`{json.dumps(hashes)}`。
- 划分：`{args.split}`；{len(rows)} 张图像、{len(captions)} 条候选描述；超过 {max_length} tokens 的 {long_count} 条文本按模型上限截断。
- 两个方向独立逐样本重算通过；第一名并列查询数分别为 {result['queries_with_top1_ties']} / {result['text_to_image_queries_with_top1_ties']}。
- 指标输出：`{output.name}`；逐样本记录：`{predictions_path.name}`、`{reverse_predictions_path.name}`；完整相似度矩阵：`{scores_path.name}`。

同图配对与跨场景结果仅作诊断，不替代主指标。该指标不代表 TICR 或极性分类准确率；本次未核验上游权重训练场景与固定测试集是否重合。
'''
    report_path.write_text(report, encoding='utf-8')
    print(report, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
