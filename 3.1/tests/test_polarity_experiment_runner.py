"""Full orchestration with real polarity scores and fake model subprocesses."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from moral_llava import polarity
from moral_llava.schema import LABEL_PAIRS
from test_polarity import audit_for, source

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_polarity_experiment.py'
spec=importlib.util.spec_from_file_location('polarity_runner_under_test',SCRIPT)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class PolarityRunnerTests(unittest.TestCase):
    def test_full_flow_independent_audit_verified_resume_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'project';data=root/'data/polarity-v1';data.mkdir(parents=True)
            counts={'train':[1211,668,1043,978,1003], 'validation':[70,42,63,58,57], 'test':[72,42,63,55,56]}
            rows_by_split={}
            for split,task_counts in counts.items():
                rows=[]
                for task,n in zip(polarity.TASKS,task_counts):
                    for i in range(n):
                        group=('A' if i<n//2 else 'B') if split=='test' else None
                        rows.extend(polarity.make_records(source(f'{split}-{group}-{i}',[LABEL_PAIRS[task][i%2]],split),group))
                polarity.write_new(data/f'{split}.json',rows);rows_by_split[split]=rows
            audit=audit_for(rows_by_split['test'])
            audit.update(source_scene_counts={'train':2138,'validation':134,'test':126},
                         question_counts={s:len(r) for s,r in rows_by_split.items()},
                         pair_counts={s:len(r)//2 for s,r in rows_by_split.items()},
                         split_record_sha256={s:polarity.sha(data/f'{s}.json') for s in counts})
            polarity.write_new(data/'audit.json',audit)
            analysis=root/'results/reference/presence-full-v1-20261103/presence_analysis.json'
            polarity.write_new(analysis,{'training_steps':9621,'selected_checkpoint_matches_final':True})
            initial=root/'models/image-lora';initial.mkdir(parents=True)
            (initial/'adapter_model.safetensors').write_bytes(b'original')
            for name in ('run_polarity.py','moral_llava/polarity.py','moral_llava/polarity_runtime.py',
                         'moral_llava/binary.py','moral_llava/binary_runtime.py','moral_llava/train.py',
                         'moral_llava/infer.py','scripts/run_polarity_experiment.py'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('frozen source fixture')
            output=Path(folder)/'experiment'

            class Finished:
                pid=999999999
                returncode=0
                def poll(self):return 0

            def popen(argv,**kwargs):
                if '-m' in argv:return Finished()
                command=argv[3]
                def option(name,default=None):return argv[argv.index(name)+1] if name in argv else default
                if command=='train':
                    destination=Path(option('--output-dir'));destination.mkdir(parents=True)
                    (destination/'adapter_model.safetensors').write_bytes(b'updated')
                    steps=int(option('--max-steps','3678'));checkpoint=destination/f'checkpoint-{steps}';checkpoint.mkdir()
                    (checkpoint/'adapter_model.safetensors').write_bytes(b'updated')
                    polarity.write_new(destination/'training_metadata.json',{'global_step':steps,'best_validation_loss':.2,
                                       'best_model_checkpoint':str(checkpoint)})
                elif command in ('infer','majority'):
                    data_path=Path(option('--data',str(data/'test.json')));rows=polarity.read(data_path)
                    if command=='majority':pred=polarity.majority_predictions(rows_by_split['train'],rows)
                    else:
                        if '--limit' in argv:rows=rows[:int(option('--limit'))]
                        pred=[]
                        for row in rows:
                            r={k:row[k] for k in ('id','scene_id','task','modality','group','caption_index')}
                            r['prompt']=f'USER: {row["conversations"][0]["value"]} ASSISTANT:'
                            pred.append(polarity.format_prediction(r,row['conversations'][1]['value']))
                    for p in pred:p.update(data_sha256=polarity.sha(data_path),decoding='fixture')
                    Path(option('--output')).write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pred))
                elif command=='score':
                    pred=[json.loads(s) for s in Path(option('--predictions')).read_text().splitlines()]
                    polarity.write_new(Path(option('--output')),polarity.score(rows_by_split['test'],pred,audit))
                else:raise AssertionError(command)
                return Finished()

            argv=[str(SCRIPT),'--output',str(output)]
            with patch.object(runner,'ROOT',root),patch.object(sys,'argv',argv),\
                 patch.object(runner.subprocess,'Popen',side_effect=popen),\
                 patch.object(runner.subprocess,'check_output',return_value='fixture-commit\n'),\
                 contextlib.redirect_stdout(io.StringIO()):runner.main()
            state=polarity.read(output/'status.json');summary=polarity.read(output/'summary.json')
            self.assertEqual(state['status'],'complete');self.assertEqual(summary['training_steps'],3678)
            self.assertTrue(summary['methods']['trained']['metrics']['eligible_for_stage2'])
            self.assertFalse(summary['methods']['majority']['metrics']['above_chance_quality_guard'])
            self.assertEqual(summary['methods']['trained']['scene_bootstrap_ticr95']['A'],[1,1])
            with patch.object(runner,'ROOT',root),patch.object(sys,'argv',argv+['--resume']),\
                 patch.object(runner.subprocess,'Popen',side_effect=AssertionError('Do not rerun verified products')),\
                 contextlib.redirect_stdout(io.StringIO()):runner.main()
            with (output/'trained.jsonl').open('a') as stream:stream.write('{}\n')
            with patch.object(runner,'ROOT',root),patch.object(sys,'argv',argv+['--resume']),\
                 patch.object(runner.subprocess,'Popen',side_effect=AssertionError('Must reject altered products')),\
                 contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'Completed stage changed'):runner.main()


if __name__=='__main__':unittest.main()
