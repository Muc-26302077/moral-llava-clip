"""Real-image CPU diagnostic; this is NOT LLaVA generation or formal TICR."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--moralclip-model', type=Path, default=Path('/root/autodl-tmp/moralclip-base'))
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/polarity-v1')
    parser.add_argument('--per-class-train', type=int, default=2)
    parser.add_argument('--per-class-validation', type=int, default=1)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.per_class_train, args.per_class_validation, args.steps) <= 0:
        parser.error('Sample counts and steps must be positive')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    started = time.monotonic()
    from moral_llava import polarity
    from moral_llava.moralclip_integration import WEIGHT_SHA256, sha
    if not args.data.exists():
        from run_polarity import main as prepare
        prepare(['prepare', '--output', str(args.data)])
    raw = ROOT / 'data/raw/SMID_moralclip'
    selected = {}
    for split, count in [('train', args.per_class_train), ('validation', args.per_class_validation)]:
        rows = polarity.load_records(args.data / f'{split}.json', raw)
        rng = random.Random(42)
        selected[split] = []
        for task in polarity.TASKS:
            for positive in (False, True):
                candidates = [r for r in rows if r['modality'] == 'image' and r['task'] == task and
                              polarity.parse_answer(r['target_json'], task) is positive]
                rng.shuffle(candidates)
                if len(candidates) < count:
                    raise ValueError('Insufficient class support')
                selected[split].extend(candidates[:count])
    if {r['scene_id'] for r in selected['train']} & {r['scene_id'] for r in selected['validation']}:
        raise ValueError('Scene split leakage')
    samples_file=args.output/'samples.json'
    if samples_file.exists() and json.loads(samples_file.read_text())!=selected:
        raise ValueError('Cannot resume CPU features with different sample identities/labels')
    write(samples_file, selected)
    source = args.moralclip_model
    if sha(source / 'model.safetensors') != WEIGHT_SHA256:
        raise ValueError('Original MoralCLIP source hash mismatch')
    write(args.output / 'status.json', {'stage': 'extracting_real_frozen_visual_features_numpy',
                                      'formal_ticr': None, 'pid': os.getpid()})
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).with_name('extract_guided_features_numpy.py')),
                    '--samples',str(args.output/'samples.json'), '--moralclip-model',str(source),
                    '--base-model',str(args.base_model), '--output',str(args.output/'features.npz')],
                   check=True, env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    # Import torch only after the bounded-memory extraction worker has exited.
    import torch
    import numpy as np
    from moral_llava.text_guided_vision import GuidanceConfig, TextGuidedResidual
    torch.set_num_threads(1)
    torch.manual_seed(42)
    mapping=json.loads((args.output/'features.json').read_text())
    with np.load(args.output/'features.npz') as cached:
        features={path:torch.from_numpy(cached['image_'+str(i)].copy())
                  for i,path in enumerate(mapping['paths'])}
        conditions={task:torch.from_numpy(cached['condition_'+task].copy()) for task in polarity.TASKS}
    unique=features
    write(args.output/'status.json',{'stage':'training_cpu_diagnostic','formal_ticr':None,'pid':os.getpid()})

    def batch(split):
        rows = selected[split]
        return (torch.stack([features[r['image_path']] for r in rows]),
                torch.stack([conditions[r['task']] for r in rows]),
                torch.tensor([polarity.TASKS.index(r['task']) for r in rows]),
                torch.tensor([int(polarity.parse_answer(r['target_json'], r['task'])) for r in rows]))

    train = batch('train')
    valid = batch('validation')
    cfg = GuidanceConfig()
    adapter = TextGuidedResidual(cfg)
    # A task-specific linear diagnostic, not the released language decoder.
    head = torch.nn.Parameter(torch.randn(5, 768, 2) * 0.001)
    bias = torch.nn.Parameter(torch.zeros(5, 2))

    def logits(x, task):
        pooled = x.mean(1)
        return torch.einsum('bd,bdk->bk', pooled, head[task]) + bias[task]

    optimizer = torch.optim.AdamW([head, bias], lr=1e-2, weight_decay=0)
    for _ in range(args.steps):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(logits(train[0], train[2]), train[3]).backward()
        optimizer.step()
    head.requires_grad_(False)
    bias.requires_grad_(False)

    @torch.no_grad()
    def measure(batch, adapted, shuffle=False):
        x, condition, task, target = batch
        if shuffle:
            condition = condition.roll(2, 0)
        output = adapter(x, condition) if adapted else x
        pred = logits(output, task)
        return {'accuracy': float((pred.argmax(-1) == target).float().mean()),
                'cross_entropy': float(torch.nn.functional.cross_entropy(pred, target))}

    baseline = {split: measure(batch(split), False) for split in selected}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-4, weight_decay=0)
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad()
        # Small CPU microbatches bound autograd activation memory.
        indices=torch.tensor([(step*2)%len(train[0]), (step*2+1)%len(train[0])])
        x, condition, task, target = [value[indices] for value in train]
        output = adapter(x, condition)
        ce = torch.nn.functional.cross_entropy(logits(output, task), target)
        preservation = (output-x).square().mean() / (x.square().mean()+1e-6)
        loss = ce + 0.05 * preservation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        losses.append({'step': step+1, 'loss': float(loss.detach()), 'cross_entropy': float(ce.detach())})
    after = {split: measure(batch(split), True) for split in selected}
    with torch.no_grad():
        visual = valid[0][:1].expand(2, -1, -1)
        changed = adapter(visual, torch.stack([conditions['care'], conditions['fairness']]))
        sensitivity = float((changed[0]-changed[1]).abs().mean())
    checkpoint = args.output / 'diagnostic-adapter.pt'
    torch.save({'adapter': adapter.state_dict(), 'head': head.detach(), 'bias': bias.detach()}, checkpoint)
    report = {'kind': 'real_moralclip_cpu_feature_diagnostic_not_llava_generation', 'seed': 42,
              'device': 'cpu', 'threads': 1, 'torch': torch.__version__, 'config': asdict(cfg),
              'train_examples': len(selected['train']), 'validation_examples': len(selected['validation']),
              'unique_images': len(unique), 'adapter_parameters': sum(p.numel() for p in adapter.parameters()),
              'baseline_linear_probe': baseline, 'guided_linear_probe': after,
              'shuffled_question_validation': measure(valid, True, shuffle=True),
              'same_image_changed_question_mean_abs_feature_difference': sensitivity,
              'adapter_losses': losses, 'formal_ticr': None,
              'seconds': time.monotonic()-started,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'vision_sha256': WEIGHT_SHA256,
              'data_sha256': {s: sha(args.data/f'{s}.json') for s in selected},
              'limitations': ['Small balanced diagnostic selected from original train/validation only.',
                             'Frozen task-specific linear probe replaces 7B decoder in this CPU diagnostic.',
                             'CPU diagnostic is not a TICR result or evidence of improved LLaVA accuracy.',
                             'Question guidance conveys known moral dimension, not scene-specific captions.']}
    write(args.output / 'cpu-diagnostic.json', report)
    lines = ['# 文本引导视觉编码：CPU 真实图像小样本诊断', '',
             '本测试使用原始 MoralCLIP 特征与真实 SMID 图像；采用冻结线性探针作为诊断输出。',
             '**未加载 7B 语言解码器，不能将本结果写成 LLaVA 准确率或 TICR。**', '',
             f'训练 {len(selected["train"])} 条，验证 {len(selected["validation"])} 条；独立图像 {len(unique)} 张。', '',
             '| 设置 | 训练准确率 | 验证准确率 | 验证交叉熵 |', '|---|---:|---:|---:|']
    for name, result in [('冻结视觉特征＋线性探针', baseline), ('文本引导残差＋相同冻结探针', after)]:
        lines.append(f'| {name} | {result["train"]["accuracy"]:.2%} | {result["validation"]["accuracy"]:.2%} | {result["validation"]["cross_entropy"]:.6f} |')
    lines += ['', f'可训练视觉残差参数：{report["adapter_parameters"]:,}。',
              f'同一图像切换问题后的平均特征差异：{sensitivity:.8g}。',
              f'耗时 {report["seconds"]:.1f} 秒；峰值进程 RSS {report["peak_rss_mib"]:.1f} MiB。',
              '结构集成、生成路径和保存重载另由微型 LLaVA 的 CPU 测试验证。正式指标等待 GPU 实测。']
    (args.output / 'cpu-diagnostic.md').write_text('\n'.join(lines)+'\n')
    write(args.output / 'status.json', {'stage': 'cpu_diagnostic_complete', 'formal_ticr': None})
    print(json.dumps({k:v for k,v in report.items() if k != 'adapter_losses'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
