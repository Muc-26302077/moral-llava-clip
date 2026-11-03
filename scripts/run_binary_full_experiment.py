"""Run the complete QA v1 first-stage experiment; suitable for detached screen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from moral_llava.binary import FOUNDATION_KEYS, TASKS, read, sha


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume a terminal interrupted run from saved checkpoints.')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.resume:
        state = read(output/'status.json')
        old_pid = state.get('runner_pid')
        if old_pid and Path(f'/proc/{old_pid}/cmdline').exists():
            command = Path(f'/proc/{old_pid}/cmdline').read_bytes()
            if b'run_binary_full_experiment.py' in command and old_pid != os.getpid():
                raise RuntimeError('The original runner is still live; do not start a duplicate.')
        child_pid = state.get('child_pid')
        if child_pid and Path(f'/proc/{child_pid}/cmdline').exists():
            command = Path(f'/proc/{child_pid}/cmdline').read_bytes()
            if b'run_binary.py' in command:
                raise RuntimeError('The original child is still live; do not restart it.')
    else:
        output.mkdir(parents=True, exist_ok=False)
        state = {'status': 'starting', 'started_at': now(), 'commands': []}
    state.update(runner_pid=os.getpid(), heartbeat_at=now(), output=str(output))
    environment = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    if args.local_files_only:
        environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    offline = ['--local-files-only'] if args.local_files_only else []
    entry = str(ROOT/'run_binary.py')
    data = ROOT/'data/binary-v1'
    initial = ROOT/'models/image-lora'
    if not initial.is_dir():
        initial = ROOT/'runs/moral-llava-1.5'

    def record():
        state['heartbeat_at'] = now()
        save(output/'status.json', state)

    def execute(name, arguments, products=()):
        previous = next((c for c in reversed(state['commands']) if c['name'] == name and c.get('returncode') == 0), None)
        if previous:
            for path, checksum in previous.get('products', {}).items():
                if not Path(path).is_file() or sha(path) != checksum:
                    raise RuntimeError(f'Completed stage changed: {name}: {path}')
            print(f'SKIP verified stage {name}', flush=True)
            return
        command = [sys.executable, '-u']+arguments
        logfile = output/f'{name}.log'
        attempt = {'name': name, 'argv': command, 'started_at': now(), 'log': str(logfile)}
        state['commands'].append(attempt)
        state.update(status='running', phase=name)
        print(f'{now()} START {name}', flush=True)
        started = time.monotonic()
        with logfile.open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            record()
            while child.poll() is None:
                time.sleep(15)
                if name == 'train':
                    with logfile.open('rb') as stream:
                        stream.seek(max(0, logfile.stat().st_size-16384))
                        tail = stream.read().decode(errors='replace')
                    matches = re.findall(r'(\d+)/(\d+) \[', tail)
                    if matches:
                        step, total = map(int, matches[-1])
                        state['training_progress'] = {'step': step, 'total': total}
                record()
        attempt.update(returncode=child.returncode, finished_at=now(), seconds=round(time.monotonic()-started, 3))
        state['child_pid'] = None
        if child.returncode:
            record()
            raise RuntimeError(f'{name} exited {child.returncode}; inspect {logfile}')
        attempt['products'] = {str(p): sha(p) for p in products}
        record()
        print(f'{now()} DONE {name} {attempt["seconds"]} seconds', flush=True)

    try:
        if not data.is_dir():
            execute('prepare', [entry, 'prepare'], [data/'audit.json'])
        audit = read(data/'audit.json')
        train, validation, test = (read(data/f'{s}.json') for s in ('train', 'validation', 'test'))
        expected_train = 2138 if (ROOT/'data/excluded_images.json').exists() else 2140
        if audit['scene_counts'] != {'train':expected_train, 'validation':134, 'test':126}:
            raise ValueError('Dataset counts differ from the fixed original protocol')
        scene_sets = [{r['scene_id'] for r in rows} for rows in (train, validation, test)]
        for split, rows, ids in zip(('train','validation','test'), (train,validation,test), scene_sets):
            if len(ids) != audit['scene_counts'][split] or len(rows) != len(ids)*12:
                raise ValueError(f'Actual record coverage differs from audit: {split}')
        if any(scene_sets[i] & scene_sets[j] for i,j in ((0,1),(0,2),(1,2))):
            raise ValueError('Scene overlap across splits')
        if any(len({r['scene_id'] for r in test if r['group']==g}) != 63 for g in ('A','B')):
            raise ValueError('Frozen A/B split is not complete')
        config = {'scope': 'QA v1 complete first stage; no automatic vision-adaptation training',
                  'epochs': 3, 'batch_size': 4, 'gradient_accumulation_steps': 2,
                  'learning_rate': 2e-5, 'seed': 42, 'precision': 'bf16',
                  'gradient_checkpointing': False, 'selection': 'minimum validation loss',
                  'initial_adapter': str(initial), 'initial_adapter_sha256': sha(initial/'adapter_model.safetensors'),
                  'scene_counts': audit['scene_counts'], 'question_counts': audit['question_counts'],
                  'data_sha256': {s:sha(data/f'{s}.json') for s in ('train','validation','test')},
                  'source_sha256': {p:sha(ROOT/p) for p in ('run_binary.py','moral_llava/binary.py','moral_llava/binary_runtime.py','moral_llava/train.py','moral_llava/infer.py')},
                  'expected_training_steps': math.ceil(len(train)/8)*3,
                  'test_policy': 'fixed configuration before baseline; no test-driven tuning; only validation-selected final adapter tested',
                  'stage2_policy': 'record eligibility, inspect disagreements; do not automatically implement stage 2'}
        config_path = output/'frozen_config.json'
        if config_path.exists():
            if read(config_path) != config:
                raise ValueError('Frozen configuration, sources, data or initialization changed')
        else:
            save(config_path, config)
            state['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        record()
        execute('unit-tests', ['-m','unittest','discover','-s','tests','-p','test_binary.py','-v'])
        smoke_data = output/'smoke-data'
        smoke_data.mkdir(exist_ok=True)
        for split, rows in (('train',train),('validation',validation)):
            ids = {r['scene_id'] for r in rows[:24]}
            subset = [r for r in rows if r['scene_id'] in ids]
            path = smoke_data/f'{split}.json'
            if not path.exists():
                save(path, subset)
        training = [entry,'train','--epochs','3','--batch-size','4','--eval-batch-size','4',
                    '--gradient-accumulation-steps','2','--learning-rate','2e-5','--seed','42',
                    '--precision','bf16','--no-gradient-checkpointing','--initial-adapter',str(initial)] + offline
        execute('gpu-smoke', training+['--train-data',str(smoke_data/'train.json'),
                '--validation-data',str(smoke_data/'validation.json'),'--max-steps','2',
                '--logging-steps','1','--output-dir',str(output/'smoke-adapter')],
                [output/'smoke-adapter/adapter_model.safetensors'])
        inference = [entry,'infer','--device','cuda','--precision','bf16','--batch-size','4','--max-new-tokens','8']+offline
        execute('smoke-reload', inference+['--data',str(smoke_data/'validation.json'),
                '--adapter',str(output/'smoke-adapter'),'--limit','12','--output',str(output/'smoke-reload.jsonl')],
                [output/'smoke-reload.jsonl'])
        execute('majority', [entry,'majority','--output',str(output/'majority.jsonl')], [output/'majority.jsonl'])
        execute('baseline-infer', inference+['--adapter',str(initial),'--output',str(output/'baseline.jsonl')], [output/'baseline.jsonl'])
        train_arguments = training+['--logging-steps','25','--output-dir',str(output/'adapter')]
        if args.resume and (output/'adapter').exists():
            checkpoints = sorted((output/'adapter').glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
            if checkpoints and not any(c['name']=='train' and c.get('returncode')==0 for c in state['commands']):
                train_arguments += ['--resume-from-checkpoint',str(checkpoints[-1])]
        execute('train', train_arguments, [output/'adapter/adapter_model.safetensors',output/'adapter/training_metadata.json'])
        execute('trained-infer', inference+['--adapter',str(output/'adapter'),'--output',str(output/'trained.jsonl')], [output/'trained.jsonl'])
        for method in ('majority','baseline','trained'):
            execute(method+'-score', [entry,'score','--predictions',str(output/f'{method}.jsonl'),
                    '--output',str(output/f'{method}-metrics.json')], [output/f'{method}-metrics.json'])
        metadata = read(output/'adapter/training_metadata.json')
        if metadata['global_step'] != config['expected_training_steps'] or not math.isfinite(metadata['best_validation_loss']):
            raise ValueError('Training did not complete the planned steps and validation selection')
        if sha(initial/'adapter_model.safetensors') != config['initial_adapter_sha256']:
            raise ValueError('Original adapter changed')
        summary = {'scope':'complete QA v1 first-stage benchmark', 'scene_counts':audit['scene_counts'],
                   'training_steps':metadata['global_step'],'best_validation_loss':metadata['best_validation_loss'],
                   'selected_checkpoint':metadata['best_model_checkpoint'],'methods':{},'created_at':now()}
        lines = ['# QA v1 完整实验结果', '',
                 f'训练 {expected_train} 场景，验证 134 场景，测试 126 场景（A/B 各 63）；每场景六项问答、两种模态。',
                 '3 epochs，BF16，batch 4，梯度累积 2，LR 2e-5，seed 42；从旧图像 LoRA 初始化，仅验证 loss 选模。',
                 '下表仅统计五个主任务，non_moral 辅助项单列在完整评分文件中。', '',
                 '| 方法 | 图片准确率 | 文字准确率 | 图文一致率 | 共同正确率 | 图片 balanced accuracy | 文字 balanced accuracy |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for method in ('majority','baseline','trained'):
            metrics = read(output/f'{method}-metrics.json')
            predictions = [json.loads(s) for s in (output/f'{method}.jsonl').read_text().splitlines() if s.strip()]
            if not metrics['complete_grouped_benchmark'] or len(predictions)!=1512:
                raise ValueError('Formal evaluation is incomplete')
            expected = {r['id']:r for r in test}
            index = {r['id']:r for r in predictions}
            if len(index)!=1512 or set(index)!=set(expected):
                raise ValueError('Prediction ID coverage is incomplete')
            answer = {k:(p['raw_prediction'].strip()=='是' if p['raw_prediction'].strip() in ('是','否') else None) for k,p in index.items()}
            macro = metrics['overall']['primary_macro']
            for modality in ('image','text'):
                accuracies, balanced = [], []
                for task in FOUNDATION_KEYS:
                    subset = [r for r in test if r['task']==task and r['modality']==modality]
                    accuracies.append(sum(answer[r['id']] is (r['conversations'][1]['value']=='是') for r in subset)/126)
                    recalls = []
                    for label in (False,True):
                        selected = [r for r in subset if (r['conversations'][1]['value']=='是') is label]
                        recalls.append(sum(answer[r['id']] is label for r in selected)/len(selected))
                    balanced.append(sum(recalls)/2)
                if not math.isclose(sum(accuracies)/5,macro[modality+'_accuracy']) or not math.isclose(sum(balanced)/5,macro[modality+'_balanced_accuracy']):
                    raise ValueError('Independent accuracy recomputation differs')
            bootstrap = {}
            for group in ('A','B'):
                scene_values = []
                for scene in sorted({r['scene_id'] for r in test if r['group']==group}):
                    rows = [r for r in test if r['scene_id']==scene and r['task'] in FOUNDATION_KEYS]
                    pairs = {t:{r['modality']:answer[r['id']] for r in rows if r['task']==t} for t in FOUNDATION_KEYS}
                    scene_values.append(sum(p['image'] is not None and p['image'] is p['text'] for p in pairs.values())/5)
                rng = random.Random(42)
                samples = sorted(sum(rng.choices(scene_values,k=63))/63 for _ in range(2000))
                bootstrap[group] = [samples[49],samples[1949]]
                if not math.isclose(sum(scene_values)/63,metrics['groups'][group]['primary_macro']['agreement']):
                    raise ValueError('Independent agreement recomputation differs')
            summary['methods'][method] = {'overall':metrics['overall'],'groups':metrics['groups'],
                    'complete_grouped_benchmark':True,'eligible_for_stage2':metrics['eligible_for_stage2'],
                    'scene_bootstrap_agreement95':bootstrap}
            fields = ('image_accuracy','text_accuracy','agreement','joint_accuracy','image_balanced_accuracy','text_balanced_accuracy')
            lines.append('| '+method+' | '+' | '.join(f'{macro[k]:.2%}' for k in fields)+' |')
        selected = read(output/'trained-metrics.json')
        lines += ['', f'训练完成 {metadata["global_step"]} 步；最佳验证 loss={metadata["best_validation_loss"]:.6f}。',
                  f'A 组一致率：{selected["groups"]["A"]["primary_macro"]["agreement"]:.2%}；B 组：{selected["groups"]["B"]["primary_macro"]["agreement"]:.2%}。',
                  f'完整基准：{selected["complete_grouped_benchmark"]}；两组匹配门槛：{selected["agreement_passes_60pct_both_groups"]}；防退化条件：{selected["above_chance_quality_guard"]}；第二阶段资格：{selected["eligible_for_stage2"]}。',
                  '资格标记仅表明预设第一阶段门槛，不承诺高共同正确率，也不自动启动视觉适配。必须结合分歧和共同错误审核。',
                  '本轮未依据测试结果调整训练配置。宏平均一致率区间按场景 bootstrap（seed 42，2000 次），详见 summary.json；不把同场景五项视作独立场景。',
                  '测试集中的 12 个场景曾用于先前小样本诊断，因此本轮完整评测不是一个全新的盲测。正式训练从原图像 LoRA 初始化，没有继续小样本权重。',
                  '所有命令、退出码、耗时见 status.json；原始回答、各任务指标和训练选模记录均在本目录。']
        summary['initial_adapter_unchanged'] = True
        summary['trained_adapter_sha256'] = sha(output/'adapter/adapter_model.safetensors')
        save(output/'summary.json', summary)
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        state.update(status='complete',phase='complete',finished_at=now(),child_pid=None,
                     eligible_for_stage2=selected['eligible_for_stage2'],report=str(output/'report.md'))
        record()
        print(f'{now()} COMPLETE {output}', flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error),finished_at=now())
        record()
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
