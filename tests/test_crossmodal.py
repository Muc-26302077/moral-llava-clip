"""Statistical/coverage regressions for paired evaluation; no model or training."""
import copy
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from moral_llava.crossmodal import (AnswerTrie, FOUNDATION_KEYS, STATES, VERSION,
                                   prepare, score, validate_protocol, wilson)


def target(state='virtue'):
    out = dict.fromkeys(FOUNDATION_KEYS, state)
    out['non_moral'] = state == 'not_annotated'
    return out


def fixture(count=2):
    scenes = [{'id': f'{g}{i}', 'group': g, 'reference': target()} for g in ('A', 'B') for i in range(count)]
    p = {'protocol': VERSION, 'threshold': .6, 'groups': {g: {'size': count} for g in ('A', 'B')},
         'scenes': scenes, 'caption_semantic_review': 'pending'}
    predictions = {name: [{'id': s['id'], 'modality': m, 'model': name, 'raw_prediction': json.dumps(target()),
                           'decoding': 'free', 'protocol_sha256': 'fixture'}
                         for s in scenes for m in ('image', 'text')] for name in ('base', 'lora')}
    return p, predictions


class CrossmodalTests(unittest.TestCase):
    def test_two_invalid_outputs_are_not_consistent_and_retention_is_undefined(self):
        p, rows = fixture()
        for records in rows.values():
            for row in records:
                row['raw_prediction'] = 'not json'
                row['prediction'] = target()  # Cached fields must be ignored.
        report = score(p, rows)
        for g in report['groups'].values():
            self.assertEqual(g['models']['lora']['strict_agreement'], 0)
            self.assertEqual(g['models']['lora']['dimension_agreement'], 0)
            self.assertIsNone(g['models']['lora']['agreement_among_valid_pairs'])
            self.assertIsNone(g['baseline_consistency_retention']['rate'])
        self.assertFalse(report['lora_passes_both_groups'])

    def test_high_constant_agreement_does_not_imply_correctness(self):
        p, rows = fixture()
        for records in rows.values():
            for row in records:
                row['raw_prediction'] = json.dumps(target('not_annotated'))
        report = score(p, rows)
        g = report['groups']['A']['models']['lora']
        self.assertEqual(g['strict_agreement'], 1)
        self.assertEqual(g['joint_correct_rate'], 0)
        self.assertEqual(g['modality_metrics']['image']['exact_match'], 0)

    def test_dimension_agreement_is_not_strict_agreement(self):
        p, rows = fixture()
        for row in rows['lora']:
            if row['modality'] == 'text':
                prediction = target()
                prediction['care'] = 'vice'
                row['raw_prediction'] = json.dumps(prediction)
        g = score(p, rows)['groups']['A']
        self.assertEqual(g['models']['lora']['strict_agreement'], 0)
        self.assertEqual(g['models']['lora']['dimension_agreement'], .8)
        self.assertEqual(g['baseline_consistency_retention']['rate'], 0)

    def test_sixty_percent_threshold_needs_38_of_63_in_each_group(self):
        p, rows = fixture(63)
        for row in rows['lora']:
            number = int(row['id'][1:])
            cutoff = 38 if row['id'].startswith('A') else 37
            if row['modality'] == 'text' and number >= cutoff:
                row['raw_prediction'] = json.dumps(target('vice'))
        report = score(p, rows)
        self.assertTrue(report['groups']['A']['models']['lora']['passes_threshold'])
        self.assertFalse(report['groups']['B']['models']['lora']['passes_threshold'])
        self.assertFalse(report['lora_passes_both_groups'])

    def test_missing_duplicate_and_mixed_protocols_are_rejected(self):
        p, rows = fixture()
        for mutate in (lambda r: r['base'].pop(), lambda r: r['base'].append(r['base'][0]),
                       lambda r: r['base'][0].update(decoding='constrained'),
                       lambda r: r['base'][0].update(protocol_sha256='different')):
            changed = copy.deepcopy(rows)
            mutate(changed)
            with self.assertRaises(ValueError):
                score(p, changed)

    def test_disjoint_deterministic_groups_and_source_label_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, processed = [], []
            for i in range(8):
                (root/f'{i}.jpg').write_bytes(f'image-{i}'.encode())
                labels = ['Care'] if i % 3 else ['non-moral']
                ref = dict.fromkeys(FOUNDATION_KEYS, 'not_annotated')
                ref['non_moral'] = labels == ['non-moral']
                if labels == ['Care']:
                    ref['care'] = 'virtue'
                original.append({'id': str(i), 'split': 'test', 'image_path': f'{i}.jpg',
                                 'captions': [f'Scene {i}', 'Unused description'], 'moral_labels': labels})
                processed.append({'id': str(i), 'conversations': [{}, {'value': json.dumps(ref)}]})
            source, converted = root/'source.json', root/'processed.json'
            source.write_text(json.dumps(original)); converted.write_text(json.dumps(processed))
            a, b = prepare(source, converted, root), prepare(source, converted, root)
            self.assertEqual(a['scenes'], b['scenes'])
            self.assertEqual([a['groups'][g]['size'] for g in ('A', 'B')], [4, 4])
            self.assertTrue(all(s['caption_index'] == 0 for s in a['scenes']))
            validate_protocol(a)
            original[0]['moral_labels'] = ['Care']
            source.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                prepare(source, converted, root)

    def test_wilson_interval_handles_zero_and_complete_successes(self):
        self.assertIsNone(wilson(0, 0))
        self.assertAlmostEqual(wilson(0, 63)[0], 0)
        self.assertAlmostEqual(wilson(63, 63)[1], 1)
        self.assertLess(wilson(38, 63)[0], .6)

    def test_trie_allows_all_243_answers_and_rejects_early_eos(self):
        class CharacterTokenizer:
            eos_token_id = 0
            pad_token_id = 1000000
            def encode(self, text, **kwargs):
                return [ord(c)+1 for c in text]
            def decode(self, ids, **kwargs):
                return ''.join(chr(i-1) for i in ids if i)
        tokenizer = CharacterTokenizer()
        trie = AnswerTrie(tokenizer)
        self.assertNotIn(0, trie.allowed([]))
        for states in product(STATES, repeat=5):
            answer = dict(zip(FOUNDATION_KEYS, states))
            answer['non_moral'] = all(s == 'not_annotated' for s in states)
            ids = tokenizer.encode(' '+json.dumps(answer, separators=(',', ':'))) + [0]
            prefix = []
            for token in ids:
                self.assertIn(token, trie.allowed(prefix))
                prefix.append(token)
            self.assertEqual(trie.allowed(prefix+[tokenizer.pad_token_id]*2), [0])
            with self.assertRaises(ValueError):
                trie.allowed(prefix+[ord('!')+1])
        with self.assertRaises(ValueError):
            trie.allowed([ord('!')+1])


if __name__ == '__main__':
    unittest.main()
