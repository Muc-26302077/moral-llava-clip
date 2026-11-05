"""Verify all delivered splits, image isolation, provenance and real evaluation evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from delivery import ROOT, save, sha, validate
from evaluate_delivery import load_delivery


def verify(parent, rebuild=True):
    parent = Path(parent).resolve()
    reports = {}
    for task in ('polarity-v1', 'legacy-three-state-v1'):
        images, scenes = {}, {}
        reports[task] = {}
        for split in ('train', 'validation', 'test'):
            root = parent/task/split
            reports[task][split] = validate(root)
            rows = load_delivery(root)
            if any(r.get('split', split) != split for r in rows):
                raise ValueError('Wrong split in source_root')
            scenes[split] = {r.get('scene_id', r['id']) for r in rows}
            manifest = json.loads((root/'manifest.json').read_text())
            images[split] = {r['sha256'] for r in manifest['files']}
        for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            if scenes[a] & scenes[b] or images[a] & images[b]:
                raise ValueError(f'Cross-split leakage: {task} {a}/{b}')
    if rebuild:
        with tempfile.TemporaryDirectory(prefix='verify-delivery-rebuild-') as tmp:
            regenerated = Path(tmp)/'delivery'
            subprocess.run([sys.executable, str(ROOT/'scripts/build_delivery.py'),
                '--config', str(parent/'config.json'), '--output', str(regenerated)],
                check=True, stdout=subprocess.DEVNULL)
            for task in reports:
                for split in reports[task]:
                    for name in ('records.jsonl', 'manifest.json'):
                        if sha(parent/task/split/name) != sha(regenerated/task/split/name):
                            raise ValueError(f'Export not reproducible: {task}/{split}/{name}')
    token_stats = json.loads((parent/'reports/current-model-token-statistics.json').read_text())
    for task in reports:
        for split in reports[task]:
            item = token_stats[task][split]
            if item['records_sha256'] != sha(parent/task/split/'records.jsonl'):
                raise ValueError('Token statistics refer to different data')
            if item['records'] != reports[task][split]['records'] or item['min_trainable_tokens'] < 1 or item['length_distribution']['max'] > item['max_seq_length']:
                raise ValueError('Token statistics budget/supervision/coverage mismatch')
    run = json.loads((parent/'reports/polarity-live-run.json').read_text())
    answers = parent/'reports/polarity-live-answers.jsonl'
    data = parent/'reports/polarity-live-evaluation-data.json'
    if run['status'] != 'complete' or run['mode'] != 'model-inference' or sha(answers) != run['answer_sha256'] or sha(data) != run['data_sha256']:
        raise ValueError('Live evaluation evidence mismatch')
    sys.path.insert(0, str(ROOT))
    from moral_llava import polarity
    original = ROOT/'results/reference/moralclip-llava-polarity-20261104'
    with tempfile.TemporaryDirectory(prefix='verify-delivery-test-') as tmp:
        audit = polarity.prepare(ROOT/'data/raw/SMID_moralclip', Path(tmp)/'reference',
                    ROOT/'results/reference/runs/crossmodal-v1/protocol.json', ROOT/'data/excluded_images.json')
        predictions = [json.loads(x) for x in answers.read_text().splitlines() if x.strip()]
        if any(p.get('data_sha256') != sha(data) or p.get('adapter_sha256') != run['adapter_sha256'] for p in predictions):
            raise ValueError('Answer provenance mismatch')
        metrics = polarity.score(json.loads(data.read_text()), predictions, audit)
        saved = json.loads((parent/'reports/polarity-live-metrics.json').read_text())
        if metrics != saved:
            raise ValueError('Saved live metrics cannot be recomputed')
    previous = {r['id']: r['raw_prediction'] for r in map(json.loads, (original/'trained.jsonl').read_text().splitlines())}
    if {r['id']: r['raw_prediction'] for r in predictions} != previous:
        raise ValueError('New input format changed model answers')
    return {'status': 'PASS', 'scope': 'project-organization-and-current-model-reproduction',
        'source_roots': 6, 'records': sum(x['records'] for v in reports.values() for x in v.values()),
        'rebuild_identical': rebuild, 'cross_split_scene_and_image_isolation': True,
        'current_model_budget_and_supervision': 'passed', 'live_answers_identical_to_original': True,
        'ticr': metrics['overall']['primary_macro']['ticr'],
        'receiver_qwen35_validation': 'pending_project_group_receiver',
        'intern_s2_evaluation': 'pending_project_group_joint_evaluation', 'format_reports': reports}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery', type=Path, default=ROOT.parent/'delivery')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = verify(args.delivery)
    save(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
