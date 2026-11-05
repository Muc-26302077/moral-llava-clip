"""Independently audit the finished presence run before any polarity training."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    project, run = args.project.resolve(), args.run.resolve()
    state = json.loads((run/'status.json').read_text())
    assert state['status'] == 'complete', state.get('error', state['status'])
    data = json.loads((project/'data/binary-v1/test.json').read_text())
    frozen = json.loads((run/'frozen_config.json').read_text())
    metadata = json.loads((run/'adapter/training_metadata.json').read_text())
    assert len(data) == 1512 and len({r['id'] for r in data}) == 1512
    assert metadata['global_step'] == 9621
    assert digest(Path(frozen['initial_adapter'])/'adapter_model.safetensors') == frozen['initial_adapter_sha256']
    for name, checksum in frozen['source_sha256'].items():
        current = project/name
        if current.exists() and digest(current) == checksum:
            continue
        # A later experiment branch may legitimately extend the runtime. Check
        # the immutable training commit rather than requiring a checkout swap.
        original = subprocess.check_output(['git','show',f'{state["source_commit"]}:3.1/{name}'],cwd=project)
        assert hashlib.sha256(original).hexdigest() == checksum, name
    selected = Path(metadata['best_model_checkpoint'])/'adapter_model.safetensors'
    assert digest(selected) == digest(run/'adapter/adapter_model.safetensors')
    tasks = ('care', 'fairness', 'loyalty', 'authority', 'sanctity')
    analysis = {'metric_name': 'TICR', 'metric_full_name': 'Text-Image Consistency Rate',
                'task': 'historical foundation presence control; NOT polarity classification',
                'training_steps': metadata['global_step'], 'best_validation_loss': metadata['best_validation_loss'],
                'selected_checkpoint': metadata['best_model_checkpoint'], 'methods': {},
                'adapter_sha256': digest(run/'adapter/adapter_model.safetensors'),
                'initial_adapter_unchanged': True, 'selected_checkpoint_matches_final': True}
    for method in ('majority', 'baseline', 'trained'):
        predictions = [json.loads(s) for s in (run/f'{method}.jsonl').read_text().splitlines() if s.strip()]
        index = {p['id']: p for p in predictions}
        assert len(predictions) == len(index) == len(data)
        assert set(index) == {r['id'] for r in data}
        values = {k: (p['raw_prediction'].strip() == '是' if p['raw_prediction'].strip() in ('是', '否') else None)
                  for k, p in index.items()}
        supplied = json.loads((run/f'{method}-metrics.json').read_text())
        grouped = {}
        for group in ('overall', 'A', 'B'):
            rows = [r for r in data if group == 'overall' or r['group'] == group]
            result = {}
            for task in tasks:
                subset = [r for r in rows if r['task'] == task]
                by_scene = {}
                for row in subset:
                    by_scene.setdefault(row['scene_id'], {})[row['modality']] = values[row['id']]
                truth = {r['scene_id']: r['conversations'][1]['value'] == '是' for r in subset}
                n = len(by_scene)
                consistent = sum(p['image'] is not None and p['image'] is p['text'] for p in by_scene.values())
                joint = sum(p['image'] is truth[s] and p['text'] is truth[s] for s, p in by_scene.items())
                score = {'pairs': n, 'ticr': consistent/n, 'joint_accuracy': joint/n,
                         'consistent_wrong': consistent-joint, 'disagreements_or_invalid': n-consistent}
                for modality in ('image', 'text'):
                    accuracy = sum(p[modality] is truth[s] for s, p in by_scene.items())/n
                    recalls = [sum(p[modality] is label for s, p in by_scene.items() if truth[s] is label) /
                               sum(v is label for v in truth.values()) for label in (False, True)]
                    score[modality+'_accuracy'] = accuracy
                    score[modality+'_balanced_accuracy'] = sum(recalls)/2
                    score[modality+'_yes_rate'] = sum(p[modality] is True for p in by_scene.values())/n
                    score[modality+'_invalid'] = sum(p[modality] is None for p in by_scene.values())
                result[task] = score
            keys = ('ticr', 'joint_accuracy', 'image_accuracy', 'text_accuracy',
                    'image_balanced_accuracy', 'text_balanced_accuracy')
            macro = {k: sum(t[k] for t in result.values())/5 for k in keys}
            reference = supplied['overall' if group == 'overall' else 'groups']
            reference = reference if group == 'overall' else reference[group]
            for key, value in macro.items():
                assert math.isclose(value, reference['primary_macro']['agreement' if key == 'ticr' else key]), (method, group, key)
            grouped[group] = {'per_task': result, 'macro': macro}
        analysis['methods'][method] = {'metrics': grouped,
                'invalid_outputs_all_tasks': sum(v is None for v in values.values()),
                'complete_grouped_benchmark': supplied['complete_grouped_benchmark'],
                'presence_protocol_pass': supplied['eligible_for_stage2']}
    trainer = json.loads((run/'adapter/trainer_state.json').read_text())
    analysis['validation_history'] = [r for r in trainer['log_history'] if 'eval_loss' in r]
    delta = {k: analysis['methods']['trained']['metrics']['overall']['macro'][k] -
                analysis['methods']['baseline']['metrics']['overall']['macro'][k]
             for k in analysis['methods']['trained']['metrics']['overall']['macro']}
    analysis['delta_vs_baseline'] = delta
    lines = ['# TICR：维度存在性微调结果分析（历史对照）', '',
             '本报告先于正式极性二分类分支建立。当前实验回答“是否涉及该维度”，正负极性均映射为“是”。它不是“关怀还是伤害”等极性任务，不能作为正式目标的达标证据。',
             '指标正式名称为 **TICR（Text-Image Consistency Rate，图文一致率）**；本报告中的 TICR 特指存在性任务。原始输出与旧评分文件保持不变，本文独立重算并核对全部 1512 条预测。', '',
             '## 训练与选模', '',
             f'训练 2138 场景 / 25656 问答，验证 134 场景 / 1608 问答，测试 126 场景 / 1512 问答。3 epochs、BF16、batch 4、梯度累积 2、LR 2e-5、seed 42，共 {metadata["global_step"]} 步。',
             f'最佳验证 loss：{metadata["best_validation_loss"]:.6f}；选中 checkpoint：`{Path(metadata["best_model_checkpoint"]).name}`。已校验最终 adapter 与该 checkpoint 权重一致，原初始化 adapter 未改变。', '',
             '| Epoch | Step | Validation loss |', '|---|---:|---:|']
    for row in analysis['validation_history']:
        lines.append(f'| {row["epoch"]:.0f} | {row["step"]} | {row["eval_loss"]:.6f} |')
    if len(analysis['validation_history']) > 1 and analysis['validation_history'][-1]['eval_loss'] > analysis['validation_history'][0]['eval_loss']:
        lines += ['', '后续轮验证损失高于第一轮，出现过拟合迹象；最终权重仍严格由最低验证损失选择，没有按测试结果挑选。']
    lines += ['', '## 完整测试结果（五主任务宏平均）', '',
              '| 方法 | 图像准确率 | 文本准确率 | TICR | 共同正确率 | 图像 balanced accuracy | 文本 balanced accuracy |',
              '|---|---:|---:|---:|---:|---:|---:|']
    keys = ('image_accuracy', 'text_accuracy', 'ticr', 'joint_accuracy', 'image_balanced_accuracy', 'text_balanced_accuracy')
    for method in ('majority', 'baseline', 'trained'):
        macro = analysis['methods'][method]['metrics']['overall']['macro']
        lines.append('| '+method+' | '+' | '.join(f'{macro[k]:.2%}' for k in keys)+' |')
    lines += ['', '## 微调后分组与分维度', '', '| 组别 | TICR | 共同正确率 |', '|---|---:|---:|']
    for group in ('A', 'B'):
        m = analysis['methods']['trained']['metrics'][group]['macro']
        lines.append(f'| {group} | {m["ticr"]:.2%} | {m["joint_accuracy"]:.2%} |')
    lines += ['', '| 维度 | 图像准确率 | 文本准确率 | TICR | 共同正确率 | 共同错误对数 |', '|---|---:|---:|---:|---:|---:|']
    for task, m in analysis['methods']['trained']['metrics']['overall']['per_task'].items():
        lines.append(f'| {task} | {m["image_accuracy"]:.2%} | {m["text_accuracy"]:.2%} | {m["ticr"]:.2%} | {m["joint_accuracy"]:.2%} | {m["consistent_wrong"]} |')
    lines += ['', '## 结论与正式极性实验', '',
              '微调的效果须同时看正确率、共同正确率与 TICR。高一致率可包含共同错误；训练集众数对照能够产生 100% 一致率，不能据此宣称模型有效。',
              f'相对原图像 LoRA 新提示基线，图像准确率变化 {delta["image_accuracy"]*100:+.2f} 个百分点，文本准确率变化 {delta["text_accuracy"]*100:+.2f} 个百分点，TICR 变化 {delta["ticr"]*100:+.2f} 个百分点。',
              f'微调后全部六项问答无效输出数：{analysis["methods"]["trained"]["invalid_outputs_all_tasks"]}/1512；存在性协议旧门槛通过情况：{analysis["methods"]["trained"]["presence_protocol_pass"]}。该门槛不代表极性分类通过。',
              '正式目标改为五个道德基础内的正负极性二分类，未标注维度不进入对应任务；保留原场景划分、原 A/B 分组及固定 caption 0。测试包含 288 个有效场景-维度对、576 条图文问答。各维度样本数不等，TICR 按五维宏平均，不能使用旧 189/315 的整数门槛。',
              '新实验从原图像 LoRA 重新初始化，不继续本轮存在性 adapter；冻结视觉塔、投影器和基座，仅训练语言 LoRA；先 smoke、原权重基线，再正式训练与验证选模，最后完整测试。',
              '历史实验已接触同一测试集，本次及后续极性评测均不能声称全新盲测；文本描述继承图像标签，未额外完成逐条文本语义重标注。不得根据这些测试结果调参。',
              '原始数据、预测及指标不覆盖；权重留在运行目录。独立重算结果见 `presence_analysis.json`。']
    (run/'presence_analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2)+'\n')
    (run/'presence_analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report': str(run/'presence_analysis.md'), 'validation': analysis['validation_history'],
                      'methods': {k: v['metrics']['overall']['macro'] for k, v in analysis['methods'].items()},
                      'delta_vs_baseline': delta}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
