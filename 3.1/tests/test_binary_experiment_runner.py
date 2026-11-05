"""Exercise full experiment orchestration and safe resume without real model weights."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from moral_llava import binary

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_binary_full_experiment.py'
spec = importlib.util.spec_from_file_location('binary_full_runner_under_test', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ExperimentRunnerTests(unittest.TestCase):
    def test_complete_reporting_verified_resume_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'project'
            data = root/'data/binary-v1'
            data.mkdir(parents=True)
            (root/'data/excluded_images.json').write_text('{}')
            initial = root/'models/image-lora'
            initial.mkdir(parents=True)
            (initial/'adapter_model.safetensors').write_bytes(b'original adapter')
            for name in ('run_binary.py','moral_llava/binary.py','moral_llava/binary_runtime.py','moral_llava/train.py','moral_llava/infer.py'):
                path = root/name
                path.parent.mkdir(exist_ok=True)
                path.write_text('source fixture')
            counts = {'train':2138,'validation':134,'test':126}
            for split,n in counts.items():
                rows=[]
                for i in range(n):
                    source = {'id':f'{split}-{i}', 'split':split, 'image_path':f'{split}/{i}.png',
                              'captions':['A scene.']*10,
                              'moral_labels':['Care','Loyalty'] if i%2 else ['Fairness','Respect','Sanctity']}
                    rows.extend(binary.make_records(source, ('A' if i<63 else 'B') if split=='test' else None))
                (data/f'{split}.json').write_text(json.dumps(rows,ensure_ascii=False))
            (data/'audit.json').write_text(json.dumps({'scene_counts':counts,'question_counts':{s:n*12 for s,n in counts.items()}}))
            output = Path(folder)/'experiment'

            class FinishedChild:
                pid = 999999999
                returncode = 0
                def poll(self): return 0

            def popen(argv, **kwargs):
                if '-m' in argv:
                    return FinishedChild()
                command = argv[3]
                def option(name, default=None):
                    return argv[argv.index(name)+1] if name in argv else default
                if command == 'train':
                    destination = Path(option('--output-dir'))
                    destination.mkdir(parents=True)
                    (destination/'adapter_model.safetensors').write_bytes(b'updated adapter')
                    steps = int(option('--max-steps', '9621'))
                    (destination/'training_metadata.json').write_text(json.dumps({'global_step':steps,'best_validation_loss':0.2,'best_model_checkpoint':'checkpoint-9621'}))
                elif command in ('infer','majority'):
                    source = Path(option('--data',str(data/'test.json')))
                    rows = binary.read(source)
                    if '--limit' in argv:
                        rows = rows[:int(option('--limit'))]
                    predictions=[]
                    for row in rows:
                        normalized = {k:row[k] for k in ('id','scene_id','task','modality','group','caption_index')}
                        normalized['prompt'] = f'USER: {row["conversations"][0]["value"]} ASSISTANT:'
                        prediction = binary.format_prediction(normalized,row['conversations'][1]['value'])
                        prediction.update(data_sha256=binary.sha(source),decoding='fixture')
                        predictions.append(prediction)
                    Path(option('--output')).write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in predictions))
                elif command == 'score':
                    predictions = [json.loads(s) for s in Path(option('--predictions')).read_text().splitlines()]
                    metrics = binary.score(binary.read(data/'test.json'),predictions)
                    Path(option('--output')).write_text(json.dumps(metrics))
                else:
                    raise AssertionError(f'Unexpected command {argv}')
                return FinishedChild()

            arguments = [str(SCRIPT),'--output',str(output)]
            with patch.object(runner,'ROOT',root), patch.object(sys,'argv',arguments), \
                 patch.object(runner.subprocess,'Popen',side_effect=popen), \
                 patch.object(runner.subprocess,'check_output',return_value='fixture-commit\n'), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            state=binary.read(output/'status.json')
            summary=binary.read(output/'summary.json')
            self.assertEqual(state['status'],'complete')
            self.assertEqual(summary['training_steps'],9621)
            self.assertTrue(summary['methods']['trained']['complete_grouped_benchmark'])
            self.assertTrue(summary['methods']['trained']['eligible_for_stage2'])
            self.assertEqual(summary['methods']['trained']['scene_bootstrap_agreement95']['A'],[1,1])
            self.assertTrue((output/'report.md').is_file())
            with patch.object(runner,'ROOT',root), patch.object(sys,'argv',arguments+['--resume']), \
                 patch.object(runner.subprocess,'Popen',side_effect=AssertionError('Verified stages must not restart')), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            with (output/'trained.jsonl').open('a') as stream:
                stream.write('{}\n')
            with patch.object(runner,'ROOT',root), patch.object(sys,'argv',arguments+['--resume']), \
                 patch.object(runner.subprocess,'Popen',side_effect=AssertionError('Must reject changed products')), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'Completed stage changed'):
                    runner.main()


if __name__=='__main__':
    unittest.main()
