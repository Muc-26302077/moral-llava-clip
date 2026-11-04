import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.evaluate_retrieval import sha256
from scripts.score_retrieval import verify


class RetrievalRescoreTests(unittest.TestCase):
    def create_fixture(self, root):
        (root / 'test').mkdir()
        rows = []
        for name, label in [('a', 'Care'), ('b', 'Care'), ('c', 'Harm')]:
            (root / 'test' / f'{name}.jpg').write_bytes(name.encode())
            rows.append({'id': name, 'split': 'test', 'image_path': f'test/{name}.jpg',
                         'captions': [name + ' caption'], 'moral_labels': [label]})
        manifest = root / 'test/smid_manifest.json'
        manifest.write_text(json.dumps(rows))
        scores = np.array([[1, 3, 0], [3, 1, 0], [3, 1, 0]], dtype=np.float32)
        predictions = []
        for q, index in zip(rows, [1, 0, 0]):
            target = rows[index]
            predictions.append({'query_id': q['id'], 'query_labels': q['moral_labels'],
                                'retrieved_image_id': target['id'], 'caption_index': 0,
                                'retrieved_caption': target['captions'][0],
                                'retrieved_labels': target['moral_labels'],
                                'r_at_1_correct': bool(set(q['moral_labels']) & set(target['moral_labels']))})
        evaluation = root / 'metrics.json'
        ev = {'metric_name': 'R@1', 'protocol': 'moral-label-r1-v1', 'original_split': 'test',
              'manifest_sha256': sha256(manifest), 'num_queries': 3, 'num_candidates': 3,
              'r_at_1': 2 / 3, 'correct_queries': 2, 'queries_with_top1_ties': 0,
              'image_sha256': {r['id']: sha256(root / r['image_path']) for r in rows},
              'metrics': {'r_at_1': 2 / 3, 'label_relevance': {'r_at_1': 2 / 3},
                          'same_image_pairing': {'r_at_1': 0}},
              'cross_image_diagnostic': {'r_at_1': 2 / 3}, 'model_weight_sha256': {'fixture': 'fixture'}}
        evaluation.write_text(json.dumps(ev))
        evaluation.with_suffix('.predictions.json').write_text(json.dumps(predictions))
        np.save(evaluation.with_suffix('.scores.npy'), scores)
        return evaluation

    def test_rescore_uses_labels_not_image_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = verify(self.create_fixture(root), root)
            self.assertEqual(result['r_at_1'], 2 / 3)
            self.assertEqual(result['paired_r_at_1_diagnostic'], 0)

    def test_tampered_labels_and_images_fail_verification(self):
        for tamper in ('label', 'image'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                evaluation = self.create_fixture(root)
                if tamper == 'label':
                    path = evaluation.with_suffix('.predictions.json')
                    rows = json.loads(path.read_text())
                    rows[0]['retrieved_labels'] = ['Harm']
                    path.write_text(json.dumps(rows))
                else:
                    (root / 'test/a.jpg').write_bytes(b'changed image')
                with self.assertRaises(ValueError):
                    verify(evaluation, root)


if __name__ == '__main__':
    unittest.main()
