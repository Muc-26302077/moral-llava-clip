"""Guard against replacing the moral-label benchmark with paired retrieval."""
import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_retrieval import load_manifest, make_parser
from moral_llava.train_alignment import make_parser as training_parser, selection_value


class RetrievalTargetTests(unittest.TestCase):
    def test_original_checkpoint_must_be_explicit(self):
        parser = make_parser()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--output', 'unused.json'])
        args = parser.parse_args(['--model', '/models/original', '--output', 'unused.json'])
        self.assertFalse(args.allow_download)
        self.assertEqual(args.split, 'test')

    def test_validation_selection_uses_moral_r1(self):
        args = training_parser().parse_args(['--output-dir', 'unused'])
        self.assertEqual(args.selection_metric, 'moral_r1')
        metrics = {'label_relevance': {'r_at_1': .8, 'map_i2t': .4},
                   'same_image_pairing': {'r_at_1': .1}}
        self.assertEqual(selection_value(metrics, args.selection_metric), .8)
        self.assertEqual(selection_value(metrics, 'paired_r1'), .1)
        self.assertEqual(selection_value(metrics, 'moral_map'), .4)

    def test_invalid_moral_labels_are_not_silently_treated_as_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            row = {'id': 'a', 'captions': ['caption'], 'moral_labels': ['non-moral']}
            path.write_text(json.dumps([row]))
            self.assertEqual(len(load_manifest(path)), 1)
            for labels in ([], ['unknown'], ['Care', 'non-moral']):
                row['moral_labels'] = labels
                path.write_text(json.dumps([row]))
                with self.assertRaises(ValueError):
                    load_manifest(path)


if __name__ == '__main__':
    unittest.main()
