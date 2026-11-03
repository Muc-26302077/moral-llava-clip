"""Five conditional moral-polarity tasks and Text-Image Consistency Rate (TICR).

Only scenes annotated at either pole enter a task. Missing annotations never
become negative examples. Original scene splits and A/B assignments are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from . import binary
from .schema import FOUNDATION_KEYS, labels_to_target
from .train import safe_image_path

VERSION = 'smid-foundation-polarity-v1'
TASKS = FOUNDATION_KEYS
ANSWERS = {'care': {True: '关怀', False: '伤害'},
           'fairness': {True: '公平', False: '欺骗'},
           'loyalty': {True: '忠诚', False: '背叛'},
           'authority': {True: '权威', False: '颠覆'},
           'sanctity': {True: '纯洁', False: '污秽'}}
read, sha, write_new = binary.read, binary.sha, binary.write_new


def parse_answer(raw, task):
    if task not in TASKS or not isinstance(raw, str):
        raise ValueError('Expected a literal polarity label for this task')
    for polarity, label in ANSWERS[task].items():
        if raw.strip() == label:
            return polarity
    raise ValueError(f'Expected only {" or ".join(ANSWERS[task].values())}; no punctuation or explanation')


def targets(labels):
    original = labels_to_target(labels)
    return {task: None if original[task] == 'not_annotated' else original[task] == 'virtue'
            for task in TASKS}


def question(task, modality, caption=None):
    if task not in TASKS or modality not in ('image', 'text'):
        raise ValueError('Unknown polarity task or modality')
    if modality == 'text' and (not isinstance(caption, str) or not caption.strip() or '<image>' in caption):
        raise ValueError('Text needs a nonempty fixed description without image tokens')
    positive, negative = ANSWERS[task][True], ANSWERS[task][False]
    subject = '图片' if modality == 'image' else '文本描述'
    query = (f'根据本数据集的标注，以下{subject}中的场景属于{positive}/{negative}道德基础维度。'
             f'请判断该场景体现的是{positive}还是{negative}。'
             f'只回答“{positive}”或“{negative}”，不要标点或解释。')
    return '<image>\n'+query if modality == 'image' else f'场景描述：\n{caption}\n\n{query}'


def make_records(source, group=None):
    truth = targets(source['moral_labels'])
    captions = source.get('captions')
    if not isinstance(captions, list) or not captions:
        raise ValueError('Each scene needs a caption')
    rows = []
    for task in TASKS:
        if truth[task] is None:
            continue
        for modality in ('image', 'text'):
            row = {'id': f'{source["id"]}::{task}::{modality}', 'scene_id': source['id'],
                   'split': source['split'], 'protocol': VERSION, 'task': task,
                   'modality': modality, 'group': group,
                   'caption_index': 0 if modality == 'text' else None,
                   'conversations': [{'from': 'human', 'value': question(task, modality, captions[0])},
                                     {'from': 'gpt', 'value': ANSWERS[task][truth[task]]}]}
            if modality == 'image':
                row['image'] = source['image_path']
            else:
                row['caption'] = captions[0]
            rows.append(row)
    return rows


def record_ids_sha(rows):
    return hashlib.sha256('\n'.join(sorted(r['id'] for r in rows)).encode()).hexdigest()


def prepare(image_root, output, groups_protocol=None, excluded_images=None):
    root, output = Path(image_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new polarity dataset directory')
    # Reuse the established source checks, including paths, source labels,
    # declared exclusions, exact-image leakage and original A/B coverage.
    with tempfile.TemporaryDirectory(prefix='ticr-source-validation-') as temporary:
        validated = binary.prepare(root, Path(temporary)/'validation', groups_protocol, excluded_images)
    paths = [root/'smid_manifest.json', root/'test/smid_manifest.json']
    sources = [r for p in paths for r in read(p)]
    excluded = set(validated['excluded_training_scene_ids'])
    groups = {r['id']: r['group'] for r in read(groups_protocol)['scenes']} if groups_protocol else {}
    rows = {s: [] for s in ('train', 'validation', 'test')}
    unannotated = {s: [] for s in rows}
    for source in sorted(sources, key=lambda r: r['id']):
        if source['id'] in excluded:
            continue
        records = make_records(source, groups.get(source['id']))
        rows[source['split']].extend(records)
        if not records:
            unannotated[source['split']].append(source['id'])
    if any(not records for records in rows.values()):
        raise ValueError('All original splits must contain annotated polarity examples')
    audit = {'protocol': VERSION, 'metric_name': 'TICR', 'metric_full_name': 'Text-Image Consistency Rate',
             'source_sha256': {str(p): sha(p) for p in paths},
             'group_protocol_sha256': validated['group_protocol_sha256'],
             'caption_policy': validated['caption_policy'],
             'label_semantics': 'annotated virtue versus vice, conditioned on task membership; unannotated excluded per task',
             'primary_tasks': list(TASKS), 'auxiliary_tasks': [],
             'source_scene_counts': validated['scene_counts'],
             'eligible_scene_counts': {s: len({r['scene_id'] for r in rs}) for s, rs in rows.items()},
             'question_counts': {s: len(rs) for s, rs in rows.items()},
             'pair_counts': {s: len(rs)//2 for s, rs in rows.items()},
             'excluded_training_scene_ids': sorted(excluded),
             'no_polarity_scene_ids': unannotated,
             'test_groups_original_scenes': dict(Counter(groups.values())),
             'test_eligible_scene_groups': {g: sorted({r['scene_id'] for r in rows['test'] if r['group'] == g}) for g in ('A', 'B')},
             'class_counts_per_modality': {s: {t: dict(Counter(r['conversations'][1]['value'] for r in rs
                    if r['task'] == t and r['modality'] == 'image')) for t in TASKS} for s, rs in rows.items()},
             'test_class_counts_by_group': {g: {t: dict(Counter(r['conversations'][1]['value'] for r in rows['test']
                    if r['group'] == g and r['task'] == t and r['modality'] == 'image')) for t in TASKS} for g in ('A', 'B')},
             'test_record_ids_sha256': record_ids_sha(rows['test']),
             'no_cross_split_exact_images': validated['no_cross_split_exact_images']}
    output.mkdir(parents=True, exist_ok=False)
    for split, records in rows.items():
        write_new(output/f'{split}.json', records)
        for task in TASKS:
            write_new(output/'tasks'/task/f'{split}.json', [r for r in records if r['task'] == task])
    audit['split_record_sha256'] = {s: sha(output/f'{s}.json') for s in rows}
    write_new(output/'audit.json', audit)
    return audit


def load_records(path, image_root, require_targets=True):
    rows, seen, result = read(path), set(), []
    if not isinstance(rows, list) or not rows:
        raise ValueError('Polarity data must be a nonempty JSON array')
    for row in rows:
        task, modality = row.get('task'), row.get('modality')
        key = row.get('id')
        if row.get('protocol') != VERSION or not isinstance(key, str) or not key or key in seen:
            raise ValueError('Invalid/duplicate polarity record or protocol')
        seen.add(key)
        if not isinstance(row.get('scene_id'), str) or not row['scene_id']:
            raise ValueError('Missing source scene ID')
        if key != f'{row["scene_id"]}::{task}::{modality}':
            raise ValueError('Record ID does not match scene, task and modality')
        human = question(task, modality, row.get('caption'))
        turns = row['conversations']
        if not turns or turns[0].get('from') != 'human' or turns[0].get('value') != human:
            raise ValueError('Polarity prompt differs from the frozen template')
        if row.get('caption_index') != (0 if modality == 'text' else None):
            raise ValueError('Caption index differs from frozen caption 0')
        answer = None
        if require_targets:
            if len(turns) != 2 or turns[1].get('from') != 'gpt':
                raise ValueError('Expected exactly one human/assistant exchange')
            answer = ANSWERS[task][parse_answer(turns[1]['value'], task)]
        result.append({'id': key, 'scene_id': row['scene_id'], 'task': task, 'modality': modality,
                       'group': row.get('group'), 'split': row['split'], 'caption_index': row.get('caption_index'),
                       'image_path': str(safe_image_path(image_root, row['image'])) if modality == 'image' else None,
                       'prompt': f'USER: {human} ASSISTANT:', 'target_json': answer})
    return result


def format_prediction(record, raw):
    output = {k: record[k] for k in ('id', 'scene_id', 'task', 'modality', 'group', 'caption_index')}
    output.update(protocol=VERSION, raw_prediction=raw, prediction=None, parse_error=None,
                  prompt_sha256=hashlib.sha256(record['prompt'].encode()).hexdigest())
    try:
        output['prediction'] = ANSWERS[record['task']][parse_answer(raw, record['task'])]
    except ValueError as error:
        output['parse_error'] = str(error)
    return output


def score(data, predictions, audit=None):
    if not data:
        raise ValueError('Empty evaluation data')
    expected, scenes, truth_map = {}, {}, {}
    for row in data:
        task, modality = row.get('task'), row.get('modality')
        if row.get('protocol') != VERSION or row['id'] in expected or task not in TASKS or modality not in ('image', 'text'):
            raise ValueError('Invalid reference metadata or duplicate ID')
        if row['id'] != f'{row["scene_id"]}::{task}::{modality}':
            raise ValueError('Reference ID differs from scene/task/modality')
        if row.get('caption_index') != (0 if modality == 'text' else None):
            raise ValueError('Reference caption index differs from fixed policy')
        key = row['scene_id'], task
        truth = parse_answer(row['conversations'][1]['value'], task)
        if key in truth_map and truth_map[key] is not truth:
            raise ValueError('Image/text references disagree')
        truth_map[key] = truth
        if row['conversations'][0]['value'] != question(task, modality, row.get('caption')):
            raise ValueError('Reference prompt differs from template')
        expected[row['id']] = row
        if row['scene_id'] in scenes and scenes[row['scene_id']] != row.get('group'):
            raise ValueError('Scene crosses groups')
        scenes[row['scene_id']] = row.get('group')
    provenance = {(r.get('decoding'), r.get('base_model_id'), r.get('base_revision'), r.get('adapter_sha256'),
                   r.get('precision'), r.get('max_new_tokens'), r.get('data_sha256')) for r in predictions}
    if len(provenance) != 1:
        raise ValueError('Mixed or missing model/decoding provenance')
    index, pairs = {}, {}
    for row in predictions:
        if row.get('smoke_only'):
            raise ValueError('Smoke predictions cannot be formally scored')
        identifier = row.get('id')
        if identifier in index or identifier not in expected:
            raise ValueError('Unexpected or duplicate prediction ID')
        ref = expected[identifier]
        for key in ('scene_id', 'task', 'modality', 'group', 'protocol', 'caption_index'):
            if row.get(key) != ref.get(key):
                raise ValueError(f'Prediction metadata mismatch: {key}')
        prompt = f'USER: {question(ref["task"], ref["modality"], ref.get("caption"))} ASSISTANT:'
        if row.get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError('Prediction prompt hash mismatch')
        try:
            answer = parse_answer(row.get('raw_prediction'), ref['task'])
        except ValueError:
            answer = None
        key = ref['scene_id'], ref['task']
        if ref['modality'] in pairs.setdefault(key, {}):
            raise ValueError('Duplicate scene/task/modality')
        pairs[key][ref['modality']] = answer
        index[identifier] = row
    if set(index) != set(expected) or any(set(p) != {'image', 'text'} for p in pairs.values()):
        raise ValueError('Missing prediction IDs or unpaired modalities')

    def aggregate(scene_ids):
        by_task = {}
        for task in TASKS:
            subset = [(truth_map[s, task], pairs[s, task]) for s in scene_ids if (s, task) in pairs]
            if not subset:
                continue
            n = len(subset)
            consistent = sum(p['image'] is not None and p['image'] is p['text'] for _, p in subset)
            joint = sum(p['image'] is truth and p['text'] is truth for truth, p in subset)
            from .crossmodal import wilson
            by_task[task] = {'pairs': n, 'consistent': consistent, 'ticr': consistent/n,
                             'joint_correct': joint, 'joint_accuracy': joint/n,
                             'consistent_wrong': consistent-joint,
                             'positive_label': ANSWERS[task][True], 'negative_label': ANSWERS[task][False],
                             'positive_references': sum(truth for truth, _ in subset),
                             'ticr_wilson95': wilson(consistent, n),
                             **{m: binary._accuracy(subset, m) for m in ('image', 'text')}}
        if not by_task:
            raise ValueError('No annotated polarity pairs')
        values = list(by_task.values())
        macro = {'ticr': sum(t['ticr'] for t in values)/len(values),
                 'joint_accuracy': sum(t['joint_accuracy'] for t in values)/len(values),
                 'consistent_pairs': sum(t['consistent'] for t in values),
                 'pairs': sum(t['pairs'] for t in values)}
        macro['micro_ticr'] = macro['consistent_pairs']/macro['pairs']
        for modality in ('image', 'text'):
            balanced = [t[modality]['balanced_accuracy'] for t in values]
            macro[modality+'_accuracy'] = sum(t[modality]['accuracy'] for t in values)/len(values)
            macro[modality+'_balanced_accuracy'] = sum(balanced)/len(balanced) if None not in balanced else None
        return {'eligible_scenes': len(scene_ids), 'per_task': by_task, 'primary_macro': macro}

    result = {'protocol': VERSION, 'metric_name': 'TICR', 'metric_full_name': 'Text-Image Consistency Rate',
              'primary_tasks': list(TASKS), 'auxiliary_tasks': [], 'overall': aggregate(list(scenes)), 'groups': {},
              'notes': ['Conditional polarity classification; unannotated scenes excluded per task.',
                        'TICR is the unweighted five-task macro mean, not pooled pair agreement.',
                        'Invalid outputs remain in denominators; two invalid answers never agree.',
                        'Test scenes and captions overlap historical experiments; this is not a fresh blind test.']}
    for group in ('A', 'B'):
        ids = [s for s, g in scenes.items() if g == group]
        if ids:
            result['groups'][group] = aggregate(ids)
    complete = bool(audit and audit.get('protocol') == VERSION and
                    {r['split'] for r in data} == {'test'} and
                    audit.get('test_groups_original_scenes') == {'A': 63, 'B': 63} and
                    audit.get('source_scene_counts', {}).get('test') == 126 and
                    record_ids_sha(data) == audit.get('test_record_ids_sha256') and
                    len(data) == audit.get('question_counts', {}).get('test') and
                    set(result['overall']['per_task']) == set(TASKS) and
                    set(result['groups']) == {'A', 'B'})
    if complete:
        for group in ('A', 'B'):
            actual_ids = sorted(s for s, g in scenes.items() if g == group)
            if actual_ids != audit['test_eligible_scene_groups'][group]:
                complete = False
            if set(result['groups'][group]['per_task']) != set(TASKS):
                complete = False
            for task in TASKS:
                actual = Counter(r['conversations'][1]['value'] for r in data
                                 if r['task'] == task and r['group'] == group and r['modality'] == 'image')
                if dict(actual) != audit['test_class_counts_by_group'][group][task]:
                    complete = False
    # Unequal per-task supports make a pooled integer threshold incorrect.
    passed = complete and all(g['primary_macro']['ticr'] >= .6 for g in result['groups'].values())
    macro = result['overall']['primary_macro']
    quality = set(result['overall']['per_task']) == set(TASKS) and all(
        macro[m+'_balanced_accuracy'] is not None and macro[m+'_balanced_accuracy'] > .5 for m in ('image', 'text'))
    result.update(complete_grouped_benchmark=complete, ticr_passes_60pct_both_groups=bool(passed),
                  above_chance_quality_guard=bool(quality), eligible_for_stage2=bool(passed and quality))
    return result


def majority_predictions(training, evaluation):
    counts = {t: Counter() for t in TASKS}
    for row in training:
        if row['split'] != 'train' or row['protocol'] != VERSION:
            raise ValueError('Majority fitting accepts polarity training records only')
        if row['modality'] == 'image':
            counts[row['task']][parse_answer(row['conversations'][1]['value'], row['task'])] += 1
    result = []
    for row in evaluation:
        task = row['task']
        if not counts[task]:
            raise ValueError('Missing training task for majority control')
        normalized = {k: row[k] for k in ('id', 'scene_id', 'task', 'modality', 'group', 'caption_index')}
        normalized['prompt'] = f'USER: {question(task, row["modality"], row.get("caption"))} ASSISTANT:'
        result.append(format_prediction(normalized, ANSWERS[task][counts[task][True] > counts[task][False]]))
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
    p.add_argument('--audit', type=Path)
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
        if any(r.get('data_sha256') != sha(args.data) for r in rows):
            raise ValueError('Prediction data hash differs from evaluation file')
        audit = read(args.audit or args.data.parent/'audit.json')
        if audit.get('split_record_sha256', {}).get('test') != sha(args.data):
            raise ValueError('Evaluation file differs from prepared complete test split')
        report = score(read(args.data), rows, audit)
        write_new(args.output, report)
    else:
        rows = majority_predictions(read(args.train_data), read(args.data))
        for row in rows:
            row.update(data_sha256=sha(args.data), decoding='constant_control')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        report = {'predictions': len(rows), 'constant_control_only': True}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
