"""Run the frozen conditional-polarity TICR experiment under detached screen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moral_llava import polarity


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def independent_audit(data, predictions, metrics):
    """Recompute metrics from raw label strings, independently of the scorer."""
    index = {r['id']: r for r in predictions}
    if len(index) != len(predictions) or set(index) != {r['id'] for r in data}:
        raise ValueError('Independent prediction coverage check failed')
    answers = {k: (p['raw_prediction'].strip() if isinstance(p['raw_prediction'], str) and
                   p['raw_prediction'].strip() in polarity.ANSWERS[p['task']].values() else None)
               for k, p in index.items()}
    bootstrap = {}
    for group in ('overall', 'A', 'B'):
        selected = [r for r in data if group == 'overall' or r['group'] == group]
        by_task = {}
        scene_pairs = {}
        for task in polarity.TASKS:
            rows = [r for r in selected if r['task'] == task]
            truth = {r['scene_id']: r['conversations'][1]['value'] for r in rows}
            pairs = {}
            for row in rows:
                pairs.setdefault(row['scene_id'], {})[row['modality']] = answers[row['id']]
            n = len(pairs)
            consistent = {s: p['image'] is not None and p['image'] == p['text'] for s, p in pairs.items()}
            ticr = sum(consistent.values())/n
            joint = sum(p['image'] == truth[s] and p['text'] == truth[s] for s, p in pairs.items())/n
            t = {'ticr': ticr, 'joint_accuracy': joint}
            for modality in ('image', 'text'):
                t[modality+'_accuracy'] = sum(p[modality] == truth[s] for s, p in pairs.items())/n
                recalls = [sum(p[modality] == label for s, p in pairs.items() if truth[s] == label)/
                           sum(v == label for v in truth.values()) for label in polarity.ANSWERS[task].values()]
                t[modality+'_balanced_accuracy'] = sum(recalls)/2
            by_task[task] = t
            for scene, value in consistent.items():
                scene_pairs.setdefault(scene, {})[task] = value
        ref = metrics['overall'] if group == 'overall' else metrics['groups'][group]
        for key in next(iter(by_task.values())):
            value = sum(t[key] for t in by_task.values())/5
            if not math.isclose(value, ref['primary_macro'][key], abs_tol=1e-12):
                raise ValueError(f'Independent recomputation differs: {group}/{key}')
        # Sample scenes together; five task denominators remain conditional and
        # unequal. Never pool all scene-task pairs as independent observations.
        rng = random.Random(42)
        scenes = sorted(scene_pairs)
        samples = []
        while len(samples) < 2000:
            draws = rng.choices(scenes, k=len(scenes))
            rates = []
            for task in polarity.TASKS:
                values = [scene_pairs[s][task] for s in draws if task in scene_pairs[s]]
                if not values:
                    break
                rates.append(sum(values)/len(values))
            if len(rates) == 5:
                samples.append(sum(rates)/5)
        samples.sort()
        bootstrap[group] = [samples[49], samples[1949]]
    return bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.resume:
        state = polarity.read(output/'status.json')
        for key, marker in (('runner_pid', b'run_polarity_experiment.py'), ('child_pid', b'run_polarity.py')):
            old_pid = state.get(key)
            path = Path(f'/proc/{old_pid}/cmdline')
            if old_pid != os.getpid() and path.exists() and marker in path.read_bytes():
                raise RuntimeError('Original experiment process is live; refusing duplicate launch')
    else:
        output.mkdir(parents=True, exist_ok=False)
        state = {'status': 'starting', 'started_at': now(), 'commands': []}
    state.update(runner_pid=os.getpid(), output=str(output))
    environment = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    if args.local_files_only:
        environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    offline = ['--local-files-only'] if args.local_files_only else []
    entry = str(ROOT/'run_polarity.py')
    data = ROOT/'data/polarity-v1'
    initial = ROOT/'models/image-lora'

    def record():
        state['heartbeat_at'] = now()
        save(output/'status.json', state)

    def execute(name, arguments, products=()):
        previous = next((c for c in reversed(state['commands']) if c['name'] == name and c.get('returncode') == 0), None)
        if previous:
            for path, checksum in previous.get('products', {}).items():
                if not Path(path).is_file() or polarity.sha(path) != checksum:
                    raise RuntimeError(f'Completed stage changed: {name}: {path}')
            print(f'SKIP verified stage {name}', flush=True)
            return
        argv = [sys.executable, '-u']+arguments
        logfile = output/f'{name}.log'
        attempt = {'name': name, 'argv': argv, 'started_at': now(), 'log': str(logfile)}
        state['commands'].append(attempt)
        state.update(status='running', phase=name)
        print(f'{now()} START {name}', flush=True)
        started = time.monotonic()
        with logfile.open('a') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            record()
            while child.poll() is None:
                time.sleep(15)
                if name == 'train':
                    with logfile.open('rb') as stream:
                        stream.seek(max(0, logfile.stat().st_size-32768))
                        tail = stream.read().decode(errors='replace')
                    steps = re.findall(r'(\d+)/'+str(state['expected_training_steps'])+r' \[', tail)
                    if steps:
                        state['training_progress'] = {'step': int(steps[-1]), 'total': state['expected_training_steps']}
                record()
        attempt.update(returncode=child.returncode, finished_at=now(), seconds=round(time.monotonic()-started, 3))
        state['child_pid'] = None
        if child.returncode:
            record()
            raise RuntimeError(f'{name} exited {child.returncode}; inspect {logfile}')
        attempt['products'] = {str(p): polarity.sha(p) for p in products}
        record()
        print(f'{now()} DONE {name}', flush=True)

    try:
        analysis_path = ROOT/'results/reference/presence-full-v1-20261103/presence_analysis.json'
        analysis = polarity.read(analysis_path)
        if analysis.get('training_steps') != 9621 or not analysis.get('selected_checkpoint_matches_final'):
            raise ValueError('The previous presence run must finish and be analyzed first')
        if not data.is_dir():
            execute('prepare', [entry, 'prepare'], [data/'audit.json'])
        audit = polarity.read(data/'audit.json')
        train, validation, test = (polarity.read(data/f'{s}.json') for s in ('train', 'validation', 'test'))
        if audit['protocol'] != polarity.VERSION or audit['source_scene_counts'] != {'train':2138, 'validation':134, 'test':126}:
            raise ValueError('Original public scene split changed')
        if audit['question_counts'] != {'train':9806, 'validation':580, 'test':576}:
            raise ValueError('Conditional-polarity coverage changed')
        for split, rows in (('train',train), ('validation',validation), ('test',test)):
            if len(rows) != audit['question_counts'][split] or polarity.sha(data/f'{split}.json') != audit['split_record_sha256'][split]:
                raise ValueError(f'Derived data differs from source audit: {split}')
        sets = [{r['scene_id'] for r in rows} for rows in (train, validation, test)]
        if any(sets[i] & sets[j] for i,j in ((0,1),(0,2),(1,2))):
            raise ValueError('Scene leakage across splits')
        sources = ('run_polarity.py', 'moral_llava/polarity.py', 'moral_llava/polarity_runtime.py',
                   'moral_llava/binary.py', 'moral_llava/binary_runtime.py', 'moral_llava/train.py',
                   'moral_llava/infer.py', 'scripts/run_polarity_experiment.py')
        config = {'protocol': polarity.VERSION, 'metric_name': 'TICR', 'metric_full_name': 'Text-Image Consistency Rate',
                  'scope': 'five conditional polarity binary tasks; no auxiliary non_moral task',
                  'epochs':3, 'batch_size':4, 'gradient_accumulation_steps':2, 'learning_rate':2e-5,
                  'seed':42, 'precision':'bf16', 'gradient_checkpointing':False,
                  'warmup_ratio':0.03, 'scheduler':'cosine', 'max_new_tokens':16,
                  'selection':'minimum validation loss', 'initial_adapter':str(initial),
                  'initial_adapter_sha256':polarity.sha(initial/'adapter_model.safetensors'),
                  'base_revision':'b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
                  'question_counts':audit['question_counts'], 'pair_counts':audit['pair_counts'],
                  'expected_training_steps':math.ceil(math.ceil(len(train)/4)/2)*3,
                  'data_sha256':{s:polarity.sha(data/f'{s}.json') for s in ('train','validation','test')},
                  'audit_sha256':polarity.sha(data/'audit.json'),
                  'source_sha256':{s:polarity.sha(ROOT/s) for s in sources},
                  'presence_analysis_sha256':polarity.sha(analysis_path),
                  'test_policy':'configuration frozen before baseline; no test-driven tuning; historical test reuse disclosed',
                  'threshold':'A/B each five-task macro TICR >= 0.60 and overall two-modality macro balanced accuracy > 0.50',
                  'stage2_policy':'report eligibility only; no automatic vision adaptation'}
        path = output/'frozen_config.json'
        if path.exists():
            if polarity.read(path) != config:
                raise ValueError('Frozen configuration, sources, analysis or data changed')
        else:
            save(path, config)
            state['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        state['expected_training_steps'] = config['expected_training_steps']
        record()
        execute('unit-tests', ['-m','unittest','discover','-s','tests','-p','test_polarity.py','-v'])
        smoke = output/'smoke-data'
        smoke.mkdir(exist_ok=True)
        for split, rows in (('train',train),('validation',validation)):
            ids = set()
            for row in rows:
                ids.add(row['scene_id'])
                if len(ids) == 4:
                    break
            path = smoke/f'{split}.json'
            if not path.exists():
                save(path, [r for r in rows if r['scene_id'] in ids])
        training = [entry,'train','--epochs','3','--batch-size','4','--eval-batch-size','4',
                    '--gradient-accumulation-steps','2','--learning-rate','2e-5','--seed','42',
                    '--precision','bf16','--no-gradient-checkpointing','--initial-adapter',str(initial)]+offline
        execute('gpu-smoke', training+['--train-data',str(smoke/'train.json'),
                '--validation-data',str(smoke/'validation.json'),'--max-steps','2',
                '--logging-steps','1','--output-dir',str(output/'smoke-adapter')],
                [output/'smoke-adapter/adapter_model.safetensors'])
        inference = [entry,'infer','--device','cuda','--precision','bf16','--batch-size','4','--max-new-tokens','16']+offline
        execute('smoke-reload', inference+['--data',str(smoke/'validation.json'),'--adapter',str(output/'smoke-adapter'),
                '--limit','12','--output',str(output/'smoke-reload.jsonl')], [output/'smoke-reload.jsonl'])
        execute('majority', [entry,'majority','--output',str(output/'majority.jsonl')], [output/'majority.jsonl'])
        execute('baseline-infer', inference+['--adapter',str(initial),'--output',str(output/'baseline.jsonl')], [output/'baseline.jsonl'])
        arguments = training+['--logging-steps','25','--output-dir',str(output/'adapter')]
        if args.resume and (output/'adapter').exists():
            checkpoints = sorted((output/'adapter').glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
            if checkpoints and not any(c['name']=='train' and c.get('returncode')==0 for c in state['commands']):
                arguments += ['--resume-from-checkpoint',str(checkpoints[-1])]
        execute('train', arguments, [output/'adapter/adapter_model.safetensors', output/'adapter/training_metadata.json'])
        execute('trained-infer', inference+['--adapter',str(output/'adapter'),'--output',str(output/'trained.jsonl')], [output/'trained.jsonl'])
        for method in ('majority','baseline','trained'):
            execute(method+'-score', [entry,'score','--predictions',str(output/f'{method}.jsonl'),
                    '--output',str(output/f'{method}-metrics.json')], [output/f'{method}-metrics.json'])
        metadata = polarity.read(output/'adapter/training_metadata.json')
        if metadata['global_step'] != config['expected_training_steps'] or not math.isfinite(metadata['best_validation_loss']):
            raise ValueError('Training did not finish the planned steps and validation selection')
        if polarity.sha(initial/'adapter_model.safetensors') != config['initial_adapter_sha256']:
            raise ValueError('Original initialization changed')
        selected = Path(metadata['best_model_checkpoint'])/'adapter_model.safetensors'
        if polarity.sha(selected) != polarity.sha(output/'adapter/adapter_model.safetensors'):
            raise ValueError('Final adapter differs from validation-selected checkpoint')
        for name, checksum in config['source_sha256'].items():
            if polarity.sha(ROOT/name) != checksum:
                raise ValueError(f'Experiment source changed while running: {name}')
        summary = {'protocol':polarity.VERSION, 'metric_name':'TICR', 'metric_full_name':'Text-Image Consistency Rate',
                   'training_steps':metadata['global_step'], 'best_validation_loss':metadata['best_validation_loss'],
                   'selected_checkpoint':metadata['best_model_checkpoint'], 'methods':{},
                   'adapter_sha256':polarity.sha(output/'adapter/adapter_model.safetensors'),
                   'initial_adapter_unchanged':True, 'selected_checkpoint_matches_final':True}
        lines = ['# TICR：五维极性二分类完整实验', '',
                 '仅纳入相应维度已标注场景，输出关怀/伤害、公平/欺骗、忠诚/背叛、权威/颠覆、纯洁/污秽；未标注不作为任一极性。',
                 '公开训练 9806 问答、验证 580 问答、测试 576 问答 / 288 个图文判断对；各维度样本数为 72/42/63/55/56。',
                 '3 epochs，BF16，batch 4，梯度累积 2，LR 2e-5，seed 42；原图像 LoRA 初始化，仅验证 loss 选模。', '',
                 '| 方法 | 图片准确率 | 文字准确率 | TICR（五维宏平均） | 共同正确率 | 图片 balanced accuracy | 文字 balanced accuracy |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for method in ('majority','baseline','trained'):
            metrics = polarity.read(output/f'{method}-metrics.json')
            if not metrics['complete_grouped_benchmark']:
                raise ValueError('Formal evaluation coverage is incomplete')
            predictions = [json.loads(s) for s in (output/f'{method}.jsonl').read_text().splitlines() if s.strip()]
            intervals = independent_audit(test, predictions, metrics)
            summary['methods'][method] = {'metrics':metrics, 'scene_bootstrap_ticr95':intervals}
            macro = metrics['overall']['primary_macro']
            fields = ('image_accuracy','text_accuracy','ticr','joint_accuracy','image_balanced_accuracy','text_balanced_accuracy')
            lines.append('| '+method+' | '+' | '.join(f'{macro[k]:.2%}' for k in fields)+' |')
        trained = summary['methods']['trained']['metrics']
        lines += ['', f'训练 {metadata["global_step"]} 步，最佳验证 loss={metadata["best_validation_loss"]:.6f}。',
                  f'A 组 TICR：{trained["groups"]["A"]["primary_macro"]["ticr"]:.2%}；B 组：{trained["groups"]["B"]["primary_macro"]["ticr"]:.2%}。',
                  f'预设一致性门槛：{trained["ticr_passes_60pct_both_groups"]}；防退化条件：{trained["above_chance_quality_guard"]}；后续视觉适配资格标记：{trained["eligible_for_stage2"]}。',
                  'TICR 包含共同错误，必须结合正确率、balanced accuracy 和共同正确率分析。各维度分母不同，不能使用旧存在性任务的 189/315 门槛。',
                  '全部原 126 测试场景均经过标签筛选；其中无任何道德基础标注的 39 场景不进入极性问答。保留的 87 场景可参与多个任务；A/B 有效场景为 43/44，判断对为 142/146。',
                  '按场景 bootstrap 2000 次、seed 42，保留同场景多维相关性，区间见 summary.json。',
                  '本次配置在基线测试前冻结；测试集曾用于历史实验，不能声称全新盲测。文本描述继承图像标签，逐条文本语义重标注未额外完成。',
                  '原始回答、命令、退出码、冻结哈希与独立复算记录全部保留；原模型未覆盖，不自动启动视觉适配。']
        save(output/'summary.json', summary)
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        state.update(status='complete', phase='complete', child_pid=None, finished_at=now(),
                     eligible_for_stage2=trained['eligible_for_stage2'], report=str(output/'report.md'))
        record()
        print(f'{now()} COMPLETE {output}', flush=True)
    except BaseException as error:
        state.update(status='failed', error=str(error), finished_at=now())
        record()
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
