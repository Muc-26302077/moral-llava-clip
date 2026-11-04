"""Independently verify a saved R@1 evaluation against its manifest and scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_retrieval import load_manifest, sha256


def verify(evaluation: Path, image_root: Path | None = None) -> dict:
    import numpy as np
    ev = json.loads(evaluation.read_text())
    root = image_root or ROOT / 'data/raw/SMID_moralclip'
    manifest = root / ('test/smid_manifest.json' if ev['original_split'] == 'test' else 'smid_manifest.json')
    if sha256(manifest) != ev['manifest_sha256']:
        raise ValueError('Manifest SHA-256 mismatch')
    rows = [r for r in load_manifest(manifest) if r['split'] == ev['original_split']]
    records = json.loads(evaluation.with_suffix('.predictions.json').read_text())
    scores = np.load(evaluation.with_suffix('.scores.npy'), allow_pickle=False)
    candidates = [(r, i, c) for r in rows for i, c in enumerate(r['captions'])]
    if (len(records) != len(rows) or scores.shape != (len(rows), len(candidates))
            or not np.isfinite(scores).all() or ev['queries_with_top1_ties']):
        raise ValueError('Invalid query count, score matrix or unresolved top-1 ties')
    correct = paired = cross = 0
    for index, (q, record) in enumerate(zip(rows, records)):
        if sha256(root / q['image_path']) != ev['image_sha256'][q['id']]:
            raise ValueError('Image SHA-256 mismatch')
        top = int(scores[index].argmax())
        target, ci, caption = candidates[top]
        relevant = bool(set(q['moral_labels']) & set(target['moral_labels']))
        if (record['query_id'] != q['id'] or set(record['query_labels']) != set(q['moral_labels'])
                or record['retrieved_image_id'] != target['id'] or record['caption_index'] != ci
                or record['retrieved_caption'] != caption or record['r_at_1_correct'] != relevant
                or set(record['retrieved_labels']) != set(target['moral_labels'])):
            raise ValueError('Saved prediction disagrees with score matrix or reference labels')
        correct += int(relevant)
        paired += int(q['id'] == target['id'])
        keep = [j for j, (r, _, _) in enumerate(candidates) if r['id'] != q['id']]
        if keep:
            cross_j = keep[int(scores[index, keep].argmax())]
            cross += int(bool(set(q['moral_labels']) & set(candidates[cross_j][0]['moral_labels'])))
    n = len(rows)
    actual = correct / n
    if (ev['metric_name'] != 'R@1' or ev['protocol'] != 'moral-label-r1-v1'
            or actual != ev['r_at_1'] or ev['correct_queries'] != correct
            or ev['num_queries'] != n or ev['num_candidates'] != len(candidates)
            or actual != ev['metrics']['label_relevance']['r_at_1']
            or actual != ev['metrics']['r_at_1']
            or paired / n != ev['metrics']['same_image_pairing']['r_at_1']
            or cross / n != ev['cross_image_diagnostic']['r_at_1']):
        raise ValueError('Metric name, objective or independent recount mismatch')
    return {'status': 'PASS', 'metric_name': 'R@1', 'protocol': ev['protocol'],
            'correct_queries': correct, 'num_queries': n, 'r_at_1': actual,
            'paired_r_at_1_diagnostic': paired / n, 'cross_image_r_at_1_diagnostic': cross / n,
            'model_weight_sha256': ev['model_weight_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--image-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.evaluation, args.image_root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
