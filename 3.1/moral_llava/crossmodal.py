"""Fixed SMID image/text groups and scoring. No model runtime is needed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path

from .evaluate import evaluate
from .schema import FOUNDATION_KEYS, STATES, TASK_PROMPT, labels_to_target, parse_prediction

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'smid-crossmodal-v1'
REVISION = 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def image_prompt():
    return f'USER: <image>\n{TASK_PROMPT} ASSISTANT:'


def text_prompt(caption):
    instruction = TASK_PROMPT.replace("this image's", "this scene's")
    return f'USER: Scene description:\n{caption}\n\n{instruction} ASSISTANT:'


def prepare(source, processed, image_root, seed=42, threshold=.60):
    if not 0 < threshold <= 1:
        raise ValueError('Threshold must be in (0,1]')
    image_root = Path(image_root).resolve()
    originals, reference_rows = read_json(source), read_json(processed)
    references = {r['id']: parse_prediction(r['conversations'][1]['value']) for r in reference_rows}
    if len(references) != len(reference_rows):
        raise ValueError('Duplicate processed IDs')
    strata, seen = defaultdict(list), set()
    for row in originals:
        identifier = row['id']
        if identifier in seen or row['split'] != 'test':
            raise ValueError('Duplicate ID or non-test scene')
        seen.add(identifier)
        target = labels_to_target(row['moral_labels'])
        if target != references.get(identifier):
            raise ValueError(f'Label mismatch: {identifier}')
        caption = row['captions'][0]
        if not isinstance(caption, str) or not caption.strip() or '<image>' in caption:
            raise ValueError(f'Invalid caption: {identifier}')
        image = (image_root / row['image_path']).resolve()
        if not image.is_relative_to(image_root) or not image.is_file():
            raise ValueError(f'Invalid image: {identifier}')
        scene = {'id': identifier, 'image': str(image), 'image_sha256': sha256(image),
                 'caption_index': 0, 'caption': caption,
                 'caption_sha256': hashlib.sha256(caption.encode()).hexdigest(),
                 'reference': target}
        strata[tuple(target[k] for k in FOUNDATION_KEYS)].append(scene)
    if seen != set(references) or len(seen) < 2:
        raise ValueError('Test ID coverage mismatch or too few scenes')
    groups = {'A': [], 'B': []}
    for _, scenes in sorted(strata.items()):
        scenes.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        for scene in scenes:
            group = min(groups, key=lambda g: len(groups[g]))
            scene['group'] = group
            groups[group].append(scene)
    return {
        'protocol': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
        'threshold': threshold, 'primary_metric': 'strict_five_foundation_agreement_all_scenes',
        'group_policy': 'disjoint test scenes; full-label strata; seeded SHA256 order; smaller group first; ties A',
        'seed': seed, 'caption_policy': 'original caption index 0 for every scene; no selection using predictions',
        'caption_semantic_review': 'pending; generated captions inherit image-level reference labels',
        'source_sha256': sha256(source), 'processed_sha256': sha256(processed),
        'base_revision': REVISION, 'image_prompt': image_prompt(),
        'text_prompt_template': text_prompt('{caption}'),
        'groups': {g: {'size': len(rows), 'minimum_consistent_pairs': math.ceil(threshold * len(rows)),
                       'label_tuple_counts': dict(Counter('|'.join(r['reference'][k] for k in FOUNDATION_KEYS) for r in rows))}
                   for g, rows in groups.items()},
        'scenes': sorted(groups['A'] + groups['B'], key=lambda r: (r['group'], r['id']))}


def validate_protocol(p):
    if p.get('protocol') != VERSION or not 0 < p['threshold'] <= 1:
        raise ValueError('Unsupported protocol')
    scenes = p['scenes']
    ids = [s['id'] for s in scenes]
    if len(ids) != len(set(ids)) or set(s['group'] for s in scenes) != {'A', 'B'}:
        raise ValueError('Invalid/overlapping groups')
    for g, definition in p['groups'].items():
        if sum(s['group'] == g for s in scenes) != definition['size']:
            raise ValueError('Group size mismatch')
    return p


class AnswerTrie:
    """Greedy token generation over all 243 legal JSON answers, without gold labels.

    This controls syntax; it is not joint maximum-likelihood answer selection.
    Free and constrained generations must be reported as different protocols.
    """
    def __init__(self, tokenizer):
        self.tree, self.max_length = {}, 0
        self.eos = tokenizer.eos_token_id
        self.pad = getattr(tokenizer, 'pad_token_id', self.eos)
        if self.eos is None:
            raise ValueError('Missing EOS')
        for values in product(STATES, repeat=5):
            target = dict(zip(FOUNDATION_KEYS, values))
            target['non_moral'] = all(v == 'not_annotated' for v in values)
            text = ' ' + json.dumps(target, separators=(',', ':'))
            ids = tokenizer.encode(text, add_special_tokens=False)
            if parse_prediction(tokenizer.decode(ids, skip_special_tokens=True)) != target:
                raise ValueError('Tokenizer round-trip failure')
            ids = list(ids) + [self.eos]
            self.max_length = max(self.max_length, len(ids))
            node = self.tree
            for token in ids:
                node = node.setdefault(int(token), {})

    def allowed(self, prefix):
        node = self.tree
        for index, token in enumerate(prefix):
            if int(token) not in node:
                raise ValueError('Generated tokens left the answer trie')
            node = node[int(token)]
            if int(token) == self.eos:
                # HF still invokes processors for finished members of a batch.
                # Their subsequent positions are padding, not generated content.
                if any(int(t) not in (self.eos, self.pad) for t in prefix[index+1:]):
                    raise ValueError('Unexpected content after EOS')
                return [self.eos]
        return sorted(node) if node else [self.eos]


def parsed(row):
    try:
        return parse_prediction(row.get('raw_prediction'))
    except (ValueError, TypeError):
        return None


def prediction_index(rows, scenes):
    expected = {(s['id'], m) for s in scenes for m in ('image', 'text')}
    result = {}
    for r in rows:
        key = (r.get('id'), r.get('modality'))
        if key in result or key not in expected:
            raise ValueError(f'Duplicate/unexpected prediction: {key}')
        result[key] = r
    if set(result) != expected:
        raise ValueError(f'Missing {len(expected - set(result))} predictions')
    return result


def wilson(successes, n):
    if n == 0:
        return None
    z, p = 1.959963984540054, successes / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0., center-half), min(1., center+half)]


def score(protocol, predictions):
    validate_protocol(protocol)
    if set(predictions) != {'base', 'lora'}:
        raise ValueError('Need base and lora predictions')
    decoding = {r.get('decoding') for rows in predictions.values() for r in rows}
    hashes = {r.get('protocol_sha256') for rows in predictions.values() for r in rows}
    if len(decoding) != 1 or len(hashes) != 1 or None in decoding or None in hashes:
        raise ValueError('Mixed or missing decoding/protocol metadata')
    scenes = protocol['scenes']
    indices = {name: prediction_index(rows, scenes) for name, rows in predictions.items()}
    report = {'protocol': VERSION, 'protocol_sha256': next(iter(hashes)), 'decoding': next(iter(decoding)),
              'threshold': protocol['threshold'], 'groups': {}, 'per_scene': [],
              'caption_semantic_review': protocol['caption_semantic_review'],
              'definitions': {
                  'strict_agreement': 'both valid and five states identical / all scenes; invalid pairs fail',
                  'dimension_agreement': 'matching states / (5 * all scenes); invalid pairs contribute zero',
                  'joint_correct': 'both outputs equal image-level reference / all scenes',
                  'retention': 'LoRA-consistent within base-consistent scenes / base-consistent scenes; null when empty',
                  'threshold_pass': 'LoRA strict agreement >= 0.60 in each group; point estimate, not CI lower bound',
                  'not_annotated': 'absence of dataset annotation; not confirmed moral neutrality'}}
    for group in ('A', 'B'):
        subset = [s for s in scenes if s['group'] == group]
        out, consistent_ids = {'num_scenes': len(subset), 'models': {}}, {}
        for model, indexed in indices.items():
            consistent, both_valid, joint = set(), 0, 0
            matches = dict.fromkeys(FOUNDATION_KEYS, 0)
            for s in subset:
                image, text = (parsed(indexed[(s['id'], m)]) for m in ('image', 'text'))
                valid = image is not None and text is not None
                equal = valid and all(image[k] == text[k] for k in FOUNDATION_KEYS)
                correct = valid and image == s['reference'] and text == s['reference']
                both_valid += int(valid)
                joint += int(correct)
                if equal:
                    consistent.add(s['id'])
                count = 0
                for k in FOUNDATION_KEYS:
                    match = int(valid and image[k] == text[k])
                    matches[k] += match
                    count += match
                report['per_scene'].append({'id': s['id'], 'group': group, 'model': model,
                    'image_prediction': image, 'text_prediction': text, 'reference': s['reference'],
                    'both_valid': valid, 'strict_agreement': bool(equal), 'both_correct': bool(correct),
                    'matching_dimensions': count})
            references = [{'id': s['id'], 'conversations': [{'from': 'gpt', 'value': json.dumps(s['reference'])}]} for s in subset]
            modality_metrics = {}
            for m in ('image', 'text'):
                full = evaluate(references, [indexed[(s['id'], m)] for s in subset])
                modality_metrics[m] = {k: full[k] for k in ('valid_output_rate', 'valid_output_count',
                    'exact_match', 'micro_f1', 'macro_f1', 'mean_foundation_accuracy', 'per_foundation',
                    'parse_failure_count', 'constraint_failure_count')}
            n = len(subset)
            consistent_ids[model] = consistent
            out['models'][model] = {'consistent_pairs': len(consistent), 'strict_agreement': len(consistent)/n,
                'strict_agreement_wilson95': wilson(len(consistent), n), 'both_valid_pairs': both_valid,
                'agreement_among_valid_pairs': len(consistent)/both_valid if both_valid else None,
                'dimension_agreement': sum(matches.values())/(5*n),
                'per_foundation_agreement': {k: v/n for k, v in matches.items()},
                'joint_correct_rate': joint/n, 'both_correct_pairs': joint,
                'modality_metrics': modality_metrics, 'passes_threshold': len(consistent)/n >= protocol['threshold']}
        denominator = len(consistent_ids['base'])
        kept = len(consistent_ids['base'] & consistent_ids['lora'])
        out['baseline_consistency_retention'] = {'numerator': kept, 'denominator': denominator,
                                                 'rate': kept/denominator if denominator else None}
        out['strict_agreement_delta_pp'] = 100*(out['models']['lora']['strict_agreement'] - out['models']['base']['strict_agreement'])
        report['groups'][group] = out
    report['lora_passes_both_groups'] = all(g['models']['lora']['passes_threshold'] for g in report['groups'].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--output', type=Path, default=ROOT/'runs/crossmodal-v1/protocol.json')
    p.add_argument('--threshold', type=float, default=.60)
    p = commands.add_parser('score')
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--base-predictions', type=Path, required=True)
    p.add_argument('--lora-predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        protocol = prepare(ROOT/'data/raw/SMID_moralclip/test/smid_manifest.json',
                           ROOT/'data/processed/test.json', ROOT/'data/raw/SMID_moralclip', threshold=args.threshold)
        write_new_json(args.output, protocol)
        print(json.dumps(protocol['groups'], ensure_ascii=False, indent=2))
    else:
        predictions = {n: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
                       for n, p in [('base', args.base_predictions), ('lora', args.lora_predictions)]}
        if any(r.get('protocol_sha256') != sha256(args.protocol) for rows in predictions.values() for r in rows):
            raise ValueError('Protocol hash mismatch')
        write_new_json(args.output, score(read_json(args.protocol), predictions))


if __name__ == '__main__':
    main()
