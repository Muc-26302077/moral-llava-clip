"""Delivery boundary tests: corruption, leakage, invalid supervision and media."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import delivery
from evaluate_delivery import load_delivery, verify_frozen


def sample():
    return {'id': 'scene::care::image', 'scene_id': 'scene', 'split': 'test',
        'protocol': 'smid-foundation-polarity-v1', 'task': 'care', 'modality': 'image',
        'group': 'A', 'caption_index': None, 'image': 'test/a.jpg',
        'conversations': [{'from': 'human', 'value': '<image>\nQuestion?'},
                          {'from': 'gpt', 'value': '关怀'}]}


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'test').mkdir()
        Image.new('RGB', (32, 16), 'white').save(self.root/'test/a.jpg')
        self.source = self.root/'source.json'
        self.source.write_text(json.dumps([sample()]))
        self.out = self.root/'out'
        delivery.export(self.source, self.root, self.out, 'test')

    def rewrite(self, obj):
        (self.out/'records.jsonl').write_text(json.dumps(obj)+'\n')

    def obj(self):
        return json.loads((self.out/'records.jsonl').read_text())

    def test_round_trip_preserves_frozen_prompt_answer_metadata_and_bytes(self):
        rows = load_delivery(self.out)
        verify_frozen(rows, [sample()])
        self.assertEqual(delivery.sha(self.out/rows[0]['image']), delivery.sha(self.root/'test/a.jpg'))
        self.assertEqual(delivery.validate(self.out)['receiver_acceptance'], 'pending_processor_validation')

    def test_frozen_boundary_rejects_prompt_target_group_and_duplicates(self):
        rows = load_delivery(self.out)
        for key, value in [('group', 'B'), ('caption_index', 1), ('conversations', [])]:
            modified = copy.deepcopy(rows)
            modified[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_frozen(modified, [sample()])
        with self.assertRaises(ValueError):
            verify_frozen(rows+rows, [sample()])

    def test_path_rejections(self):
        for name in ('../a.jpg', '/a.jpg', 'assets/../a.jpg', 'a..jpg/../b.jpg',
                     'assets/.tmp/a.jpg', 'assets/~a.jpg', 'assets/a b.jpg',
                     'assets/图.jpg', 'https://a/a.jpg', 'data:image/png;base64,aaa',
                     'assets\\a.jpg', 'assets//a.jpg'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                delivery.media_path(self.out, name)

    def test_corrupt_checksum_dimensions_missing_and_duplicate_manifest(self):
        manifest = json.loads((self.out/'manifest.json').read_text())
        for key, value in [('sha256', '0'*64), ('bytes', 1), ('width', 100), ('height', 17)]:
            changed = copy.deepcopy(manifest)
            changed['files'][0][key] = value
            delivery.save(self.out/'manifest.json', changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                delivery.validate(self.out)
        changed = copy.deepcopy(manifest)
        changed['files'] *= 2
        delivery.save(self.out/'manifest.json', changed)
        with self.assertRaises(ValueError):
            delivery.validate(self.out)
        (self.out/'manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):
            delivery.validate(self.out)

    def test_unregistered_media_and_encoding_mismatch(self):
        Image.new('RGB', (2, 2)).save(self.out/'extra.png')
        with self.assertRaises(ValueError):
            delivery.validate(self.out)
        (self.out/'extra.png').unlink()
        name = json.loads((self.out/'manifest.json').read_text())['files'][0]['path']
        Image.new('RGB', (2, 2)).save(self.out/name, format='PNG')
        with self.assertRaises(ValueError):
            delivery.validate(self.out)

    def test_symlink_escape_and_unrelated_source_files(self):
        (self.out/'leak.py').write_text('pass')
        with self.assertRaises(ValueError):
            delivery.validate(self.out)
        (self.out/'leak.py').unlink()
        (self.out/'linked').symlink_to(self.root/'test', target_is_directory=True)
        with self.assertRaises(ValueError):
            delivery.validate(self.out)

    def test_video_and_nonuser_images_rejected(self):
        original = self.obj()
        for role in ('system', 'assistant', 'tool', 'developer'):
            obj = copy.deepcopy(original)
            obj['trace']['messages'][0]['role'] = role
            self.rewrite(obj)
            with self.subTest(role=role), self.assertRaises(ValueError):
                delivery.validate(self.out)
        obj = copy.deepcopy(original)
        obj['trace']['messages'][0]['content'][0] = {'type': 'video', 'video': 'assets/a.mp4'}
        self.rewrite(obj)
        with self.assertRaises(ValueError):
            delivery.validate(self.out)

    def test_only_last_real_query_has_supervision(self):
        obj = {'messages': [{'role': 'user', 'content': 'first'},
                            {'role': 'assistant', 'content': 'history'},
                            {'role': 'user', 'content': 'last'}]}
        with self.assertRaises(ValueError):
            delivery.record(obj)
        obj['messages'].append({'role': 'assistant', 'content': 'last answer', 'use_loss': False})
        with self.assertRaises(ValueError):
            delivery.record(obj)
        obj['messages'][-1]['use_loss'] = True
        delivery.record(obj)
        obj['messages'][-1]['use_loss'] = 'false'
        with self.assertRaises(ValueError):
            delivery.record(obj)

    def test_trace_does_not_fallback_and_tool_arguments_must_be_objects(self):
        with self.assertRaises(ValueError):
            delivery.record({'trace': None, 'messages': [{'role': 'user', 'content': 'q'}]})
        obj = {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': '',
                'tool_calls': [{'type': 'function', 'function': {'name': 'search', 'arguments': '{"q":1}'}}]}]}
        delivery.record(obj)
        for bad in ('[]', '3', '{oops', None, []):
            obj['messages'][1]['tool_calls'][0]['function']['arguments'] = bad
            with self.subTest(arguments=bad), self.assertRaises(ValueError):
                delivery.record(obj)

    def test_hidden_jsonl_is_not_imported(self):
        (self.out/'.private').mkdir()
        (self.out/'.private/ignore.jsonl').write_text('bad JSON')
        self.assertEqual(delivery.validate(self.out)['records'], 1)

    def test_export_refuses_missing_images_and_overwrite(self):
        with self.assertRaises(FileExistsError):
            delivery.export(self.source, self.root, self.out, 'again')
        (self.root/'test/a.jpg').unlink()
        with self.assertRaises(FileNotFoundError):
            delivery.export(self.source, self.root, self.root/'new', 'missing')
        self.assertFalse((self.root/'new').exists())


if __name__ == '__main__':
    unittest.main()
