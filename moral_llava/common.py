"""Source checks and statistics shared by the final polarity pipeline."""
import hashlib
import json
import math
from pathlib import Path
from .schema import labels_to_target
from .train import safe_image_path

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

def wilson(successes, n):
    if n == 0:
        return None
    z, p = 1.959963984540054, successes / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0., center-half), min(1., center+half)]

def validate_sources(image_root, groups_protocol=None, excluded_images=None):
    root = Path(image_root).resolve()
    paths = [root/'smid_manifest.json', root/'test/smid_manifest.json']
    sources = [row for path in paths for row in read(path)]
    excluded = {r['id']: r for r in read(excluded_images)['images']} if excluded_images else {}
    groups = {}
    if groups_protocol:
        for row in read(groups_protocol)['scenes']:
            if row['id'] in groups or row['group'] not in ('A', 'B'):
                raise ValueError('Duplicate or invalid frozen group')
            groups[row['id']] = row['group']
        if set(groups) != {r['id'] for r in sources if r['split'] == 'test'}:
            raise ValueError('Frozen groups do not exactly cover the test scenes')
    identifiers, image_paths, hashes, skipped = set(), set(), {}, []
    counts = dict.fromkeys(('train','validation','test'), 0)
    for source in sorted(sources, key=lambda r: r['id']):
        identifier, split = source['id'], source['split']
        if identifier in identifiers or split not in counts:
            raise ValueError('Duplicate scene or invalid split')
        identifiers.add(identifier)
        labels_to_target(source['moral_labels'])
        captions = source.get('captions')
        if not isinstance(captions,list) or not captions or not all(isinstance(c,str) and c.strip() for c in captions):
            raise ValueError('Every scene needs nonempty captions')
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
        counts[split] += 1
    if set(skipped) != set(excluded) or not all(counts.values()):
        raise ValueError('Invalid exclusions or empty source split')
    return {'excluded_training_scene_ids': skipped, 'scene_counts': counts,
            'no_cross_split_exact_images': True,
            'group_protocol_sha256': sha(groups_protocol) if groups_protocol else None,
            'caption_policy': 'fixed caption 0; no label-conditioned caption selection'}
