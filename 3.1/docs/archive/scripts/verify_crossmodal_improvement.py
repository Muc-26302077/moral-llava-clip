"""Independently recount raw answers and check paired-training provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moral_llava.crossmodal import sha256
from moral_llava.schema import FOUNDATION_KEYS, parse_prediction


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    training = ROOT / 'runs/crossmodal-joint-v1'
    new_run = ROOT / 'runs/crossmodal-joint-v1-test'
    old_run = ROOT / 'runs/crossmodal-v1/full'
    evaluation = ROOT / 'outputs/crossmodal-joint-v1-evaluation'
    protocol_path = ROOT / 'runs/crossmodal-v1/protocol.json'
    protocol, config = read(protocol_path), read(training / 'run_config.json')
    status, selection = read(new_run / 'status.json'), read(training / 'selection.json')
    adapter = ROOT / selection['selected_adapter'] / 'adapter_model.safetensors'
    manifest_path = ROOT / 'data/raw/SMID_moralclip/smid_manifest.json'
    checks = {
        'training_complete': read(training / 'status.json')['status'] == 'complete',
        'inference_complete': status['status'] == 'complete',
        'protocol_unchanged': sha256(protocol_path) == read(old_run / 'status.json')['protocol_sha256'] == status['protocol_sha256'],
        'training_source_unchanged': sha256(ROOT / 'moral_llava/train_crossmodal.py') == config['source_sha256'],
        'initial_adapter_unchanged': sha256(ROOT / 'runs/moral-llava-1.5/adapter_model.safetensors') == config['initial_adapter_sha256'],
        'manifest_unchanged': sha256(manifest_path) == config['manifest_sha256'],
        'selected_adapter_loaded': sha256(adapter) == status['adapter_sha256'] and
            (ROOT / status['arguments']['adapter']).resolve() == adapter.parent.resolve(),
        'adapter_weights_changed': sha256(adapter) != config['initial_adapter_sha256'],
    }
    validation = {
        name: read(training / (name + '.json'))['metrics']
        for name in ['validation_initial', 'validation_epoch_001', 'validation_epoch_002', 'validation_epoch_003']
    }
    initial = validation['validation_initial']
    best_epoch, best = 0, initial['strict_agreement']
    for epoch in range(1, 4):
        m = validation[f'validation_epoch_{epoch:03d}']
        eligible = (m['image_exact_match'] + m['text_exact_match']) / 2 >= (initial['image_exact_match'] + initial['text_exact_match']) / 2 - config['quality_tolerance'] and m['joint_correct_rate'] >= initial['joint_correct_rate'] - config['quality_tolerance']
        if eligible and m['strict_agreement'] > best:
            best_epoch, best = epoch, m['strict_agreement']
    checks['validation_selection_reproduced'] = best_epoch == selection['selected_epoch']
    manifest = read(manifest_path)
    train_validation_ids = {r['id'] for r in manifest}
    image_root = manifest_path.parent
    train_validation_images = {sha256(image_root / r['image_path']) for r in manifest}
    overlap = {
        'ids': sorted(s['id'] for s in protocol['scenes'] if s['id'] in train_validation_ids),
        'exact_images': sorted(s['id'] for s in protocol['scenes'] if s['image_sha256'] in train_validation_images),
    }
    checks['test_ids_and_images_disjoint'] = not any(overlap.values())
    expected = {(s['id'], m) for s in protocol['scenes'] for m in ('image', 'text')}
    indices, coverage, free_equal = {}, {}, {}
    summaries, group_summaries = {}, {}
    for version, directory in [('old', old_run), ('new', new_run)]:
        for decoding in ('free', 'constrained'):
            rows = [json.loads(line) for line in (directory / f'lora_{decoding}.jsonl').read_text().splitlines()]
            assert len(rows) == len(expected) == 252
            index = {}
            for row in rows:
                key = row['id'], row['modality']
                assert key not in index
                index[key] = parse_prediction(row['raw_prediction'])
                assert row['protocol_sha256'] == sha256(protocol_path)
                scene = next(s for s in protocol['scenes'] if s['id'] == row['id'])
                prompt = protocol['image_prompt'] if row['modality'] == 'image' else protocol['text_prompt_template'].replace('{caption}', scene['caption'])
                assert row['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
            assert set(index) == expected
            indices[version, decoding] = index
            coverage[f'{version}_{decoding}'] = len(rows)
        free_equal[version] = indices[version, 'free'] == indices[version, 'constrained']
        index = indices[version, 'constrained']
        all_counts = Counter()
        by_group = {}
        for group in ('A', 'B'):
            counts = Counter()
            for s in protocol['scenes']:
                if s['group'] != group:
                    continue
                a, b = [index[s['id'], m] for m in ('image', 'text')]
                equal, correct = a == b, a == b == s['reference']
                counts.update({'scenes': 1, 'consistent': int(equal), 'joint_correct': int(correct),
                               'consistent_wrong': int(equal and not correct),
                               'image_correct': int(a == s['reference']), 'text_correct': int(b == s['reference']),
                               'both_all_not_annotated': int(a['non_moral'] and b['non_moral'])})
                for k in FOUNDATION_KEYS:
                    if a[k] != b[k]:
                        counts['presence_disagreements' if 'not_annotated' in (a[k], b[k]) else 'polarity_disagreements'] += 1
            all_counts.update(counts)
            by_group[group] = dict(counts)
        summaries[version] = dict(all_counts)
        group_summaries[version] = by_group
        report = read((old_run if version == 'old' else evaluation) / 'metrics_constrained.json')
        for group, c in by_group.items():
            reported = report['groups'][group]['models']['lora']
            assert c['consistent'] == reported['consistent_pairs']
            assert c['joint_correct'] == reported['both_correct_pairs']
            assert c['image_correct'] / c['scenes'] == reported['modality_metrics']['image']['exact_match']
            assert c['text_correct'] / c['scenes'] == reported['modality_metrics']['text']['exact_match']
    gains, distributions = {}, {}
    for group in ('A', 'B'):
        counts = Counter()
        for s in protocol['scenes']:
            if s['group'] != group:
                continue
            a, b = [indices['old', 'constrained'][s['id'], m] for m in ('image', 'text')]
            x, y = [indices['new', 'constrained'][s['id'], m] for m in ('image', 'text')]
            if a != b and x == y:
                counts['gained_correct' if x == s['reference'] else 'gained_wrong'] += 1
        gains[group] = dict(counts)
    for version in ('old', 'new'):
        for modality in ('image', 'text'):
            counter = Counter(tuple(indices[version, 'constrained'][s['id'], modality][k] for k in FOUNDATION_KEYS) for s in protocol['scenes'])
            distributions[f'{version}_{modality}'] = {'unique_label_tuples': len(counter), 'largest_tuple_count': max(counter.values())}
    assert all(checks.values()), checks
    output = {'status': 'PASS', 'checks': checks, 'split_overlap': overlap,
              'coverage': coverage, 'free_equals_constrained': free_equal,
              'validation_metrics': validation, 'overall': summaries, 'groups': group_summaries,
              'gains': gains, 'prediction_diversity': distributions,
              'selected_adapter_sha256': sha256(adapter), 'protocol_sha256': sha256(protocol_path)}
    with args.output.open('x') as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'PASS', 'checks': checks, 'overall': summaries}, indent=2))


if __name__ == '__main__':
    main()
