"""Exercise released entry points across working directories and moved layouts."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from moral_llava import pipeline
from moral_llava import download_model


class PortableEntryTests(unittest.TestCase):
    def test_python_scores_from_another_working_directory_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='moral portability ') as folder:
            root = Path(folder)
            output = root/'nested output'/'scores'
            command = [sys.executable, str(pipeline.ROOT/'run.py'), 'score', '--output', str(output)]
            first = subprocess.run(command,cwd=root,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            for decoding in ['constrained','free']:
                actual = json.loads((output/f'metrics_{decoding}.json').read_text())
                expected = json.loads((pipeline.REFERENCE/f'evaluation/metrics_{decoding}.json').read_text())
                self.assertEqual(actual,expected)
            before = (output/'metrics_constrained.json').read_bytes()
            second = subprocess.run(command,cwd=root,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual((output/'metrics_constrained.json').read_bytes(),before)

    @unittest.skipUnless(shutil.which('sh'),'POSIX shell unavailable')
    def test_moved_shell_wrapper_preserves_spaces_and_arguments(self):
        with tempfile.TemporaryDirectory(prefix='moral shell ') as folder:
            root = Path(folder)
            copied = root/'moved project with spaces'
            copied.mkdir()
            shutil.copyfile(pipeline.ROOT/'run.sh',copied/'run.sh')
            (copied/'run.py').write_text('import json,sys;print(json.dumps(sys.argv[1:]))\n')
            arguments = ['score','--output',str(root/'output with spaces')]
            result = subprocess.run(['sh',str(copied/'run.sh'),*arguments],cwd=root,
                                    env={**os.environ,'PYTHON':sys.executable},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),arguments)

    def test_default_outputs_belong_to_the_section_and_do_not_collide(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(pipeline,'ROOT',Path(folder)):
                first = pipeline.output_directory('score')
                second = pipeline.output_directory('score')
            self.assertEqual(first.parent,Path(folder)/'results/runs')
            self.assertNotEqual(first,second)

    def test_offline_download_uses_pinned_revision_and_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory(prefix='moral pinned ') as folder:
            snapshot = Path(folder)/pipeline.REVISION
            snapshot.mkdir()
            expected = {}
            for number in range(3):
                name = f'weight-{number}.safetensors'
                content = f'original weight {number}'.encode()
                (snapshot/name).write_bytes(content)
                expected[name] = {'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
            downloader = Mock(return_value=str(snapshot))
            with patch.object(download_model,'EXPECTED_WEIGHTS',expected),patch.dict(
                    sys.modules,{'huggingface_hub':SimpleNamespace(snapshot_download=downloader)}):
                actual,checks = pipeline.resolve_base(offline=True)
                self.assertEqual(actual,snapshot)
                self.assertTrue(all(c['ok'] for c in checks))
                self.assertEqual(downloader.call_args.kwargs['revision'],pipeline.REVISION)
                self.assertTrue(downloader.call_args.kwargs['local_files_only'])
                (snapshot/'weight-0.safetensors').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'verification failed'):
                    pipeline.resolve_base(offline=True)


if __name__=='__main__':
    unittest.main()
