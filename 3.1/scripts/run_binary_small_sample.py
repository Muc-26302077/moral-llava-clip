"""Run a fixed, explicitly non-formal QA v1 GPU experiment without overwrites."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moral_llava.binary import FOUNDATION_KEYS, TASKS, read, sha


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/binary-v1')
    parser.add_argument('--sample-manifest', type=Path,
                        default=ROOT/'results/reference/binary-small-v1/manifest.json')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    output = args.output or ROOT/'results/runs'/('binary-small-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output = output.resolve()
    adapter = ROOT/'models/image-lora'
    if not adapter.exists():
        adapter = ROOT/'runs/moral-llava-1.5'
    frozen = read(args.sample_manifest) if args.sample_manifest.is_file() else None
    offline = ['--local-files-only'] if args.local_files_only else []
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data').mkdir()
    environment = dict(os.environ, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    if args.local_files_only:
        environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    rng = random.Random(42)
    manifest = {'scope': 'small_sample_diagnostic_only', 'formal_benchmark': False,
                'seed': 42, 'sampling': 'sorted scene IDs shuffled, without inspecting labels; test 6 scenes per frozen group',
                'output': str(output), 'commands': [], 'splits': {},
                'initial_adapter_sha256': sha(adapter/'adapter_model.safetensors')}
    if frozen:
        manifest['sampling'] = 'reuse archived scene IDs; no new label- or prediction-based selection'
    for split, count in [('train', 32), ('validation', 8), ('test', 12)]:
        source = args.data_dir/f'{split}.json'
        rows = read(source)
        ids = sorted({r['scene_id'] for r in rows})
        if split == 'test':
            ids = []
            for group in ('A', 'B'):
                candidates = sorted({r['scene_id'] for r in rows if r['group'] == group})
                rng.shuffle(candidates)
                ids.extend(candidates[:count//2])
        else:
            rng.shuffle(ids)
            ids = ids[:count]
        chosen = set(frozen['splits'][split]['scene_ids'] if frozen else ids)
        if not chosen <= {r['scene_id'] for r in rows}:
            raise ValueError('Archived sample IDs are not available in this dataset')
        selected = [r for r in rows if r['scene_id'] in chosen]
        assert len(selected) == count*12
        assert {r['task'] for r in selected} == set(TASKS)
        destination = output/'data'/f'{split}.json'
        dump(destination, selected)
        manifest['splits'][split] = {'scene_ids': sorted(ids), 'scenes': count,
                                    'records': len(selected), 'source_sha256': sha(source),
                                    'subset_sha256': sha(destination)}
    dump(output/'manifest.json', manifest)
    print(f'OUTPUT={output}', flush=True)

    def run(name, arguments):
        command = [sys.executable] + arguments
        print(f'START {name}', flush=True)
        started = time.monotonic()
        with (output/f'{name}.log').open('w') as log:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        result = {'name': name, 'argv': command, 'returncode': completed.returncode,
                  'seconds': round(time.monotonic()-started, 3)}
        manifest['commands'].append(result)
        dump(output/'manifest.json', manifest)
        if completed.returncode:
            print((output/f'{name}.log').read_text()[-6000:], flush=True)
            raise RuntimeError(f'{name} failed; see {output}')
        print(f'DONE {name} {result["seconds"]}s', flush=True)

    run('unit-tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_binary.py', '-v'])
    evaluation = str(output/'data/test.json')
    run('majority', ['run_binary.py', 'majority', '--data', evaluation,
                     '--output', str(output/'majority.jsonl')])
    run('majority-score', ['run_binary.py', 'score', '--data', evaluation,
                           '--predictions', str(output/'majority.jsonl'), '--output', str(output/'majority-metrics.json')])
    inference = ['run_binary.py', 'infer', '--data', evaluation, '--device', 'cuda',
                 '--precision', 'bf16', '--batch-size', '4', '--max-new-tokens', '8'] + offline
    run('baseline-infer', inference + ['--adapter', str(adapter), '--output', str(output/'baseline.jsonl')])
    run('baseline-score', ['run_binary.py', 'score', '--data', evaluation,
                           '--predictions', str(output/'baseline.jsonl'), '--output', str(output/'baseline-metrics.json')])
    run('train', ['run_binary.py', 'train', '--train-data', str(output/'data/train.json'),
                  '--validation-data', str(output/'data/validation.json'), '--output-dir', str(output/'adapter'),
                  '--epochs', '1', '--batch-size', '4', '--eval-batch-size', '4',
                  '--gradient-accumulation-steps', '2', '--learning-rate', '2e-5',
                  '--logging-steps', '1', '--no-gradient-checkpointing', '--initial-adapter', str(adapter)] + offline)
    run('trained-infer', inference + ['--adapter', str(output/'adapter'), '--output', str(output/'trained.jsonl')])
    run('trained-score', ['run_binary.py', 'score', '--data', evaluation,
                          '--predictions', str(output/'trained.jsonl'), '--output', str(output/'trained-metrics.json')])
    assert sha(adapter/'adapter_model.safetensors') == manifest['initial_adapter_sha256']
    metadata = read(output/'adapter/training_metadata.json')
    assert metadata['global_step'] == 48 and metadata['best_validation_loss'] is not None
    summary = {'scope': '12-scene diagnostic; not a formal benchmark or stage-2 approval',
               'unit_tests': '15 passed', 'training_steps': metadata['global_step'],
               'validation_loss': metadata['best_validation_loss'], 'initial_adapter_unchanged': True, 'methods': {}}
    lines = ['# QA v1 小样本实测', '', '固定 seed 42；训练 32 场景（384 条），验证 8 场景（96 条），测试 12 场景（144 条，A/B 各 6）。',
             '按场景随机抽样，不依据标签或预测筛选。测试文件覆盖全部六项与两种模态，评分仅用于小样本诊断。', '',
             '| 方法 | 图片准确率 | 文字准确率 | 图文一致率 | 共同正确率 | 图片 balanced accuracy | 文字 balanced accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method in ('majority', 'baseline', 'trained'):
        metrics = read(output/f'{method}-metrics.json')
        assert metrics['complete_grouped_benchmark'] is False and metrics['eligible_for_stage2'] is False
        predictions = [json.loads(line) for line in (output/f'{method}.jsonl').read_text().splitlines()]
        assert len(predictions) == 144 and len({p['id'] for p in predictions}) == 144
        macro = metrics['overall']['primary_macro']
        fields = ('image_accuracy', 'text_accuracy', 'agreement', 'joint_accuracy', 'image_balanced_accuracy', 'text_balanced_accuracy')
        primary = [p for p in predictions if p['task'] in FOUNDATION_KEYS]
        valid = {m: sum(p['raw_prediction'].strip() in ('是', '否') for p in primary if p['modality'] == m)/60 for m in ('image', 'text')}
        summary['methods'][method] = {'primary_macro': macro, 'validity': valid,
                                      'groups': {g: v['primary_macro'] for g, v in metrics['groups'].items()}}
        labels = {'majority': '训练集众数', 'baseline': '旧图像 LoRA + 新提示', 'trained': '小样本 SFT 1 轮'}
        lines.append('| '+labels[method]+' | '+' | '.join('不可计算' if macro[k] is None else f'{macro[k]:.2%}' for k in fields)+' |')
    lines += ['', f'专项单元测试 15 项通过；真实 7B 训练完成 {metadata["global_step"]} 步，验证 loss={metadata["best_validation_loss"]:.6f}，保存后重载并完成推理。',
              '训练只更新语言 LoRA，视觉塔和投影器冻结；原 adapter 哈希未变。',
              '表格仅统计五个主任务，non_moral 辅助项及逐任务有效率、F1、A/B 结果见各 metrics.json。',
              '小样本不构成正式达标结论；未按本次测试结果追加调参。完整 126 场景验证及第二阶段视觉适配仍未执行。',
              '', '全部命令、退出状态、耗时、抽样 ID 与输入哈希见 manifest.json；原始回答见 baseline.jsonl 和 trained.jsonl。']
    dump(output/'summary.json', summary)
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    manifest['status'] = 'complete'
    dump(output/'manifest.json', manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
