"""Scene-preserving binary foundation QA, preparation and CPU-only scoring.

Presence merges the two annotated poles; it cannot recover virtue/vice.
No model dependencies are imported by this module's preparation/scoring CLI.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import FOUNDATION_KEYS, labels_to_target
from .train import safe_image_path

VERSION = 'smid-binary-foundation-v1'
TASKS = FOUNDATION_KEYS + ('non_moral',)
ANSWERS = {True: '是', False: '否'}
NAMES = {'care': '关怀/伤害（Care/Harm）', 'fairness': '公平/欺骗（Fairness/Cheating）',
         'loyalty': '忠诚/背叛（Loyalty/Betrayal）', 'authority': '权威/颠覆（Respect/Subversion）',
         'sanctity': '纯洁/污秽（Sanctity/Degradation）'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def parse_answer(raw):
    if not isinstance(raw, str) or raw.strip() not in ('是', '否'):
        raise ValueError('Expected only 是 or 否, without explanation or punctuation')
    return raw.strip() == '是'


def targets(labels):
    original = labels_to_target(labels)
    return {**{k: original[k] != 'not_annotated' for k in FOUNDATION_KEYS},
            'non_moral': original['non_moral']}


def question(task, modality, caption=None):
    if task not in TASKS or modality not in ('image', 'text'):
        raise ValueError('Unknown binary task or modality')
    if modality == 'text' and (not isinstance(caption, str) or not caption.strip() or '<image>' in caption):
        raise ValueError('Text needs a nonempty description without image tokens')
    subject = '图片' if modality == 'image' else '文本描述'
    query = (f'根据本数据集的标注，以下{subject}中的场景是否被标为 non-moral（五个道德维度均未标注）？'
             if task == 'non_moral' else
             f'根据本数据集的标注，以下{subject}中的场景是否涉及{NAMES[task]}维度？任一极性被标注即为是。')
    query += '未标注不表示现实中的道德中性。只回答是或否，不要解释。'
    return '<image>\n' + query if modality == 'image' else f'场景描述：\n{caption}\n\n{query}'


def make_records(source, group=None):
    target = targets(source['moral_labels'])
    captions = source.get('captions')
    if not isinstance(captions, list) or not captions:
        raise ValueError('Every scene must have a caption')
    rows = []
    for task in TASKS:
        for modality in ('image', 'text'):
            row = {'id': f'{source["id"]}::{task}::{modality}', 'scene_id': source['id'],
                   'split': source['split'], 'protocol': VERSION, 'task': task,
                   'modality': modality, 'group': group, 'caption_index': 0 if modality == 'text' else None,
                   'conversations': [{'from': 'human', 'value': question(task, modality, captions[0])},
                                     {'from': 'gpt', 'value': ANSWERS[target[task]]}]}
            if modality == 'image':
                row['image'] = source['image_path']
            else:
                row['caption'] = captions[0]
            rows.append(row)
    return rows


def prepare(image_root, output, groups_protocol=None, excluded_images=None):
    root, output = Path(image_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new binary dataset directory')
    paths = [root/'smid_manifest.json', root/'test/smid_manifest.json']
    sources = [row for p in paths for row in read(p)]
    identifiers, hashes, image_paths = set(), {}, set()
    excluded = {r['id']: r for r in read(excluded_images)['images']} if excluded_images else {}
    groups = {}
    if groups_protocol:
        frozen = read(groups_protocol)
        for row in frozen['scenes']:
            if row['id'] in groups or row['group'] not in ('A', 'B'):
                raise ValueError('Duplicate or invalid frozen group')
            groups[row['id']] = row['group']
        if set(groups) != {r['id'] for r in sources if r['split'] == 'test'}:
            raise ValueError('Frozen groups do not exactly cover the test scenes')
    rows = {split: [] for split in ('train', 'validation', 'test')}
    skipped = []
    for source in sorted(sources, key=lambda r: r['id']):
        identifier, split = source['id'], source['split']
        if identifier in identifiers or split not in rows:
            raise ValueError('Duplicate scene or invalid split')
        identifiers.add(identifier)
        targets(source['moral_labels'])
        if identifier in excluded:
            if split != 'train' or source['image_path'] != excluded[identifier]['image_path']:
                raise ValueError('Exclusions must match explicitly declared training images')
            skipped.append(identifier)
            continue
        path = safe_image_path(root, source['image_path'])
        if Path(source['image_path']).parts[0] != split or path in image_paths:
            raise ValueError('Image path/split mismatch or reused image path')
        image_paths.add(path)
        digest = sha(path)
        if digest in hashes and hashes[digest] != split:
            raise ValueError('Image bytes overlap across splits')
        hashes[digest] = split
        rows[split].extend(make_records(source, groups.get(identifier)))
    if set(skipped) != set(excluded):
        raise ValueError('Declared exclusion IDs absent from source manifest')
    if any(not value for value in rows.values()):
        raise ValueError('All original splits must be nonempty')
    audit = {'protocol': VERSION, 'source_sha256': {str(p): sha(p) for p in paths},
             'group_protocol_sha256': sha(groups_protocol) if groups_protocol else None,
             'caption_policy': 'fixed caption 0; no label-conditioned caption selection',
             'label_semantics': 'presence of either annotated pole; no polarity recovery; absent is not verified neutral',
             'primary_tasks': list(FOUNDATION_KEYS), 'auxiliary_tasks': ['non_moral'],
             'excluded_training_scene_ids': skipped,
             'scene_counts': {s: len(r)//12 for s, r in rows.items()},
             'question_counts': {s: len(r) for s, r in rows.items()},
             'class_counts_per_modality': {s: {t: dict(Counter(x['conversations'][1]['value'] for x in r
                   if x['task'] == t and x['modality'] == 'image')) for t in TASKS} for s, r in rows.items()},
             'test_groups': dict(Counter(groups.values())), 'no_cross_split_exact_images': True}
    # Source validation is complete before any derived directory is created.
    output.mkdir(parents=True, exist_ok=False)
    for split, records in rows.items():
        write_new(output/f'{split}.json', records)
        for task in TASKS:
            write_new(output/'tasks'/task/f'{split}.json', [r for r in records if r['task'] == task])
    write_new(output/'audit.json', audit)
    return audit


def load_records(path, image_root, require_targets=True):
    """Inference reads only human input and metadata, never assistant answers."""
    rows, seen, result = read(path), set(), []
    if not isinstance(rows, list) or not rows:
        raise ValueError('Binary data must be a nonempty JSON array')
    for row in rows:
        key = row.get('id')
        if not isinstance(key, str) or not key or key in seen or row.get('protocol') != VERSION:
            raise ValueError('Invalid/duplicate binary record or protocol')
        seen.add(key)
        task, modality = row['task'], row['modality']
        human = question(task, modality, row.get('caption'))
        turns = row['conversations']
        if not turns or turns[0].get('from') != 'human' or turns[0].get('value') != human:
            raise ValueError('Binary prompt differs from the frozen template')
        if not isinstance(row.get('scene_id'), str) or not row['scene_id']:
            raise ValueError('Missing source scene ID')
        answer = None
        if require_targets:
            if len(turns) != 2 or turns[1].get('from') != 'gpt':
                raise ValueError('Expected one human/assistant exchange')
            answer = ANSWERS[parse_answer(turns[1]['value'])]
        result.append({'id': key, 'scene_id': row['scene_id'], 'task': task, 'modality': modality,
                       'group': row.get('group'), 'split': row['split'],
                       'caption_index': row.get('caption_index'),
                       'image_path': str(safe_image_path(image_root, row['image'])) if modality == 'image' else None,
                       'prompt': f'USER: {human} ASSISTANT:', 'target_json': answer})
    return result


def format_prediction(record, raw):
    output = {k: record[k] for k in ('id', 'scene_id', 'task', 'modality', 'group', 'caption_index')}
    output.update(protocol=VERSION, raw_prediction=raw, prediction=None, parse_error=None,
                  prompt_sha256=hashlib.sha256(record['prompt'].encode()).hexdigest())
    try:
        output['prediction'] = parse_answer(raw)
    except ValueError as error:
        output['parse_error'] = str(error)
    return output


def _accuracy(pairs, modality):
    values = [(truth, answers[modality]) for truth, answers in pairs]
    n = len(values)
    recall = [sum(pred is truth for truth, pred in values if truth == label) /
              sum(truth == label for truth, _ in values) if any(truth == label for truth, _ in values) else None
              for label in (False, True)]
    tp = sum(truth and pred is True for truth, pred in values)
    fp = sum(not truth and pred is True for truth, pred in values)
    fn = sum(truth and pred is not True for truth, pred in values)
    return {'accuracy': sum(pred is truth for truth, pred in values)/n,
            'balanced_accuracy': sum(recall)/2 if None not in recall else None,
            'positive_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'valid_output_rate': sum(pred is not None for _, pred in values)/n,
            'positive_predictions': sum(pred is True for _, pred in values),
            'invalid_outputs': sum(pred is None for _, pred in values),
            'constant_prediction': len({pred for _, pred in values}) == 1}


def score(data, predictions):
    expected, scenes, truth_map = {}, {}, {}
    if not data:
        raise ValueError('Empty evaluation data')
    for row in data:
        if row['protocol'] != VERSION or row['id'] in expected or row['task'] not in TASKS or row['modality'] not in ('image', 'text'):
            raise ValueError('Invalid reference metadata or duplicate ID')
        key = row['scene_id'], row['task']
        truth = parse_answer(row['conversations'][1]['value'])
        if key in truth_map and truth_map[key] is not truth:
            raise ValueError('Image/text references disagree')
        truth_map[key] = truth
        human = question(row['task'], row['modality'], row.get('caption'))
        if row['conversations'][0]['value'] != human:
            raise ValueError('Reference prompt differs from template')
        expected[row['id']] = row
        if row['scene_id'] in scenes and scenes[row['scene_id']] != row.get('group'):
            raise ValueError('Scene crosses groups')
        scenes[row['scene_id']] = row.get('group')
    index, pairs = {}, {}
    provenance = {(r.get('decoding'), r.get('base_model_id'), r.get('base_revision'), r.get('adapter_sha256'),
                   r.get('precision'), r.get('max_new_tokens'))
                  for r in predictions}
    if len(provenance) > 1:
        raise ValueError('Mixed model or decoding provenance')
    for row in predictions:
        if row.get('smoke_only'):
            raise ValueError('Smoke predictions cannot be used for formal metrics')
        identifier = row.get('id')
        if identifier in index or identifier not in expected:
            raise ValueError('Unexpected or duplicate prediction ID')
        ref = expected[identifier]
        for k in ('scene_id', 'task', 'modality', 'group', 'protocol', 'caption_index'):
            if row.get(k) != ref.get(k):
                raise ValueError(f'Prediction metadata mismatch: {k}')
        prompt = f'USER: {question(ref["task"], ref["modality"], ref.get("caption"))} ASSISTANT:'
        if row.get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError('Prediction prompt hash mismatch')
        try:
            answer = parse_answer(row.get('raw_prediction'))
        except ValueError:
            answer = None
        index[identifier] = row
        key = ref['scene_id'], ref['task']
        if ref['modality'] in pairs.setdefault(key, {}):
            raise ValueError('Duplicate scene/task/modality')
        pairs[key][ref['modality']] = answer
    if set(index) != set(expected) or any(set(p) != {'image', 'text'} for p in pairs.values()):
        raise ValueError('Missing prediction IDs or unpaired modalities')
    used_tasks = set(t for _, t in pairs)
    if any({t for s, t in pairs if s == scene} != used_tasks for scene in scenes):
        raise ValueError('Each scene must cover the same tasks')

    def aggregate(scene_ids):
        by_task = {}
        for task in TASKS:
            subset = [(truth_map[s, task], pairs[s, task]) for s in scene_ids if (s, task) in pairs]
            if not subset:
                continue
            n = len(subset)
            consistent = sum(p['image'] is not None and p['image'] is p['text'] for _, p in subset)
            joint = sum(p['image'] is truth and p['text'] is truth for truth, p in subset)
            by_task[task] = {'pairs': n, 'consistent': consistent, 'agreement': consistent/n,
                             'joint_correct': joint, 'joint_accuracy': joint/n,
                             'consistent_wrong': consistent-joint,
                             'positive_references': sum(truth for truth, _ in subset),
                             **{m: _accuracy(subset, m) for m in ('image', 'text')}}
            from .crossmodal import wilson
            by_task[task]['agreement_wilson95'] = wilson(consistent, n)
            by_task[task]['joint_accuracy_wilson95'] = wilson(joint, n)
        primary = [by_task[t] for t in FOUNDATION_KEYS if t in by_task]
        if not primary:
            raise ValueError('At least one primary foundation is required')
        macro = {'agreement': sum(t['agreement'] for t in primary)/len(primary),
                 'joint_accuracy': sum(t['joint_accuracy'] for t in primary)/len(primary),
                 'consistent_pairs': sum(t['consistent'] for t in primary),
                 'pairs': sum(t['pairs'] for t in primary)}
        macro['minimum_consistent_pairs_60pct'] = (3*macro['pairs']+4)//5
        for modality in ('image', 'text'):
            balanced = [t[modality]['balanced_accuracy'] for t in primary]
            macro[modality+'_accuracy'] = sum(t[modality]['accuracy'] for t in primary)/len(primary)
            macro[modality+'_balanced_accuracy'] = sum(balanced)/len(balanced) if None not in balanced else None
        full = sum(all(pairs[s,t]['image'] is not None and pairs[s,t]['image'] is pairs[s,t]['text']
                       for t in FOUNDATION_KEYS) for s in scene_ids) if set(FOUNDATION_KEYS) <= used_tasks else None
        joint_full = sum(all(pairs[s,t]['image'] is truth_map[s,t] and pairs[s,t]['text'] is truth_map[s,t]
                             for t in FOUNDATION_KEYS) for s in scene_ids) if full is not None else None
        return {'scenes': len(scene_ids), 'per_task': by_task, 'primary_macro': macro,
                'five_presence_complete_agreement': full/len(scene_ids) if full is not None else None,
                'five_presence_joint_exact': joint_full/len(scene_ids) if joint_full is not None else None}

    output = {'protocol': VERSION, 'primary_tasks': list(FOUNDATION_KEYS), 'auxiliary_tasks': ['non_moral'],
              'overall': aggregate(list(scenes)), 'groups': {},
              'notes': ['Presence merges virtue and vice; not comparable to original three-state exact match.',
                        'Two invalid answers never count as consistent. Cached predictions are ignored.',
                        'Agreement alone cannot establish correct moral classification.']}
    for group in ('A', 'B'):
        ids = [s for s, g in scenes.items() if g == group]
        if ids:
            output['groups'][group] = aggregate(ids)
    complete = ({r['split'] for r in data} == {'test'} and
                set(FOUNDATION_KEYS) <= used_tasks and set(output['groups']) == {'A', 'B'} and
                len(scenes) == 126 and all(g['scenes'] == 63 for g in output['groups'].values()))
    passed = complete and all(g['primary_macro']['consistent_pairs'] >=
                             g['primary_macro']['minimum_consistent_pairs_60pct']
                             for g in output['groups'].values())
    quality = all(output['overall']['primary_macro'][m+'_balanced_accuracy'] is not None and
                  output['overall']['primary_macro'][m+'_balanced_accuracy'] > .5 for m in ('image', 'text'))
    output.update(complete_grouped_benchmark=complete, agreement_passes_60pct_both_groups=bool(passed),
                  above_chance_quality_guard=quality, eligible_for_stage2=bool(passed and quality))
    return output


def majority_predictions(training, evaluation):
    """A train-only constant control; no test-label-dependent choice."""
    counts = {t: Counter() for t in TASKS}
    for r in training:
        if r['split'] != 'train':
            raise ValueError('Majority fitting accepts training records only')
        if r['modality'] == 'image':
            counts[r['task']][parse_answer(r['conversations'][1]['value'])] += 1
    if any(not counts[r['task']] for r in evaluation):
        raise ValueError('Missing training task for majority control')
    result = []
    for r in evaluation:
        c = counts[r['task']]
        value = c[True] > c[False]  # Deterministic tie: no.
        normalized = {k: r[k] for k in ('id','scene_id','task','modality','group','caption_index')}
        normalized['prompt'] = f'USER: {question(r["task"],r["modality"],r.get("caption"))} ASSISTANT:'
        result.append(format_prediction(normalized, ANSWERS[value]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups-protocol', type=Path)
    p.add_argument('--excluded-images', type=Path)
    p = commands.add_parser('score')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('majority')
    p.add_argument('--train-data', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        report = prepare(args.image_root, args.output, args.groups_protocol, args.excluded_images)
    elif args.command == 'score':
        rows = [json.loads(s) for s in args.predictions.read_text().splitlines() if s.strip()]
        checksum = sha(args.data)
        if any(r.get('data_sha256') != checksum for r in rows):
            raise ValueError('Prediction data hash differs from the evaluation file')
        report = score(read(args.data), rows)
        write_new(args.output, report)
    else:
        rows = majority_predictions(read(args.train_data), read(args.data))
        checksum = sha(args.data)
        for row in rows:
            row.update(data_sha256=checksum, decoding='constant_control')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            for r in rows:
                handle.write(json.dumps(r, ensure_ascii=False)+'\n')
        report = {'predictions': len(rows), 'constant_control_only': True}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
