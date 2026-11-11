"""User-facing entry points must resolve the repo outside its working directory."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_documented_entrypoint_help_from_another_directory(self):
        entries=[['run_retrieval.py','--help'],['run_text_guided_vision.py','--help']]
        entries += [['run_moral_llava.py',command,'--help'] for command in ('prepare','train','infer','score','majority')]
        entries += [['scripts/'+name+'.py','--help'] for name in ('download_models','train_baseline',
            'evaluate_benchmark','evaluate_moralise','verify_results','verify_repository','delivery','evaluate_delivery')]
        with tempfile.TemporaryDirectory() as cwd:
            for script,*arguments in entries:
                with self.subTest(script=script,arguments=arguments):
                    result=subprocess.run([sys.executable,str(ROOT/script)]+arguments,cwd=cwd,
                                          capture_output=True,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':
    unittest.main()
