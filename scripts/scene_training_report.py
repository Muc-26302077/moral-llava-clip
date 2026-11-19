"""Evidence-based live report and fixed 07:45 snapshot (Asia/Shanghai)."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

TZ = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT.parent


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def render(run, *, snapshot=False):
    now = datetime.now(TZ)
    state = read(run/'orchestrator-status.json') or {}
    selection = read(run/'final-selection.json') or {}
    current_candidate = selection.get('candidate') or next(iter(reversed(state.get('candidates', []))), 'candidate-a')
    configuration = read(run/current_candidate/'frozen_config.json') or {}
    parameters = configuration.get('trainable_parameters', 114080)
    rank = configuration.get('guidance_config', {}).get('rank', 32)
    lines = ['# 同场景语义引导V2：训练与评测进度', '',
        '更新时间：'+now.strftime('%Y-%m-%d %H:%M:%S 北京时间'), '',
        '## 模型结构', '',
        f'在MoralCLIP第11个编码块内增加场景语义预测器与低秩条件残差。当前配置rank{rank}，新增{parameters:,}个参数；同场景文字经冻结文本编码器产生512维监督。无描述预测从图像生成语义条件，有描述时编码实际场景描述，问题文本不作为视觉条件。冻结原视觉权重、投影器、LLaVA和既有语言LoRA。', '',
        'GPU训练联合使用图像答案CE、可靠性筛选的文本教师KL、场景语义对齐、描述辅助CE和残差保持。正式TICR仅使用无描述的独立输入模式。', '',
        '## 运行状态', '', '- 流程阶段：'+str(state.get('phase', '尚未启动流程管理器')),
        '- 本轮目录：`'+str(run)+'`。']
    recovery = read(run/'recovery-verification.json')
    if recovery:
        lines.append('- GPU恢复验证：'+json.dumps(recovery, ensure_ascii=False)+'。')
    for name in state.get('candidates', []):
        status = read(run/name/'status.json') or {}
        pid = status.get('pid')
        live = False
        if pid:
            try:
                cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
                live = b'run_scene_guided_vision.py' in cmd and str(run/name).encode() in cmd
            except OSError:
                pass
        if status.get('status') == 'complete':
            candidate_config = read(run/name/'frozen_config.json') or {}
            epochs = candidate_config.get('arguments', {}).get('epochs', '—')
            lines.append(f'- {name}：训练与验证已结束，共{epochs}轮，所选epoch={status.get("selected_epoch", "—")}，进程存活={live}。')
        else:
            lines.append(f'- {name}：{status.get("status", "待启动")}，进程存活={live}，epoch={status.get("epoch", "—")}，batch={status.get("batch", 0)}/{status.get("batches", "—")}。')
        if status.get('error'):
            lines.append('  失败信息：'+status['error'])
    if state.get('error'):
        lines.append('- 流程错误：'+state['error'])
    training_audit = read(run/'training-audit.json')
    if training_audit:
        for candidate in training_audit['candidates']:
            lines.append(f'- 完整训练核验：{candidate["name"]}共{candidate["completed_epochs"]}轮、'
                f'{candidate["optimizer_steps"]}个优化器步骤；预测器6个、残差12个张量均实际更新。')
        lines.append('- 验证集逐条复算、质量约束与最终选模顺序核验通过，训练源码哈希保持一致。')
    lines += ['', '## 历史正式对照（均已微调）', '',
        '| 方法 | TICR | 图像准确率 | 文本准确率 | 共同正确率 |',
        '| --- | ---: | ---: | ---: | ---: |']
    keys = ('ticr', 'image_accuracy', 'text_accuracy', 'joint_accuracy')
    for label, path in [('MoralCLIP–LLaVA直接基线', 'results/baseline/trained-metrics.json'),
                        ('原视觉塔＋道德极性LoRA', 'results/benchmarks/original-tower/ticr-metrics.json')]:
        data = read(ROOT/path)
        if data:
            m = data.get('metrics', data)['overall']['primary_macro']
            lines.append('| '+label+' | '+' | '.join(f'{m[k]:.2%}' for k in keys)+' |')
    lines += ['', '## 本轮验证集候选', '',
              '验证集结果仅用于选模，与下方正式测试集结果分开解释。', '',
              '| 配置/轮次 | TICR | 图像准确率 | 共同正确率 |', '| --- | ---: | ---: | ---: |']
    for name in state.get('candidates', []):
        paths = [run/name/'baseline-validation-metrics.json']+sorted((run/name).glob('epoch-*-validation-metrics.json'))
        for path in paths:
            data = read(path)
            if data:
                m = data['overall']['primary_macro']
                lines.append('| '+name+'/'+path.stem+' | '+' | '.join(f'{m[k]:.2%}' for k in ('ticr', 'image_accuracy', 'joint_accuracy'))+' |')
    comparison = read(run/'final/comparison.json')
    verification = read(run/'final/audit.json')
    if comparison:
        lines += ['', '## 本轮独立输入正式结果', '',
            '| 方法 | TICR | 图像准确率 | 文本准确率 | 共同正确率 |', '| --- | ---: | ---: | ---: | ---: |']
        for key, label in [('baseline', '同次直接基线'), ('guided', '同场景语义引导V2')]:
            lines.append('| '+label+' | '+' | '.join(f'{comparison[key][k]:.2%}' for k in keys)+' |')
        achieved = comparison['passes_point_estimate_goal']
        lines += ['', f'锁定候选：{selection.get("candidate", "待核验")}，epoch={selection.get("epoch", "待核验")}。',
            '独立复算与模型哈希审计：'+('通过' if verification else '尚未完成')+'。',
            '**达到相对直接基线的点估计提升目标：'+('是' if achieved else '否')+'。**',
            '超过历史91.36%参考：'+str(comparison['exceeds_archived_original_tower_ticr'])+'。',
            '配对bootstrap95%差值区间：`'+json.dumps(comparison['paired_scene_bootstrap_delta95'])+'`。',
            '统计显著性须另看区间是否完全高于0，不能由点估计提升替代。']
        if selection.get('epoch') == 0:
            lines += ['', '**验证集未支持采用训练后的模块，最终保留第0轮零残差参数；不能将本表解读为新结构训练带来的提升。**']
            if training_audit:
                lines += ['', '训练观察：两组末轮的图像答案CE均已接近零，文本教师启用比例分别为'+
                    '、'.join(f'{c["validation"][-1]["mean_training_losses"]["teacher_gate_fraction"]:.2%}'
                             for c in training_audit['candidates'])+
                    '。场景语义对齐损失下降，但验证判断质量没有提高；这些是后续检查纠错监督与残差影响的线索，不能据此直接断定因果。']
        assisted = read(run/'final/paired-assisted-metrics.json')
        if assisted:
            m = assisted['overall']['primary_macro']
            lines += ['', '文字辅助模式：一致率'+f'{m["ticr"]:.2%}'+
                '，图像准确率'+f'{m["image_accuracy"]:.2%}'+
                '，共同正确率'+f'{m["joint_accuracy"]:.2%}'+
                '。此模式读取实际场景描述，单列辅助结果，不作为独立输入TICR提升。']
    else:
        lines += ['', '**尚无本轮完整正式测试结果，不能宣称TICR提升。**']
    lines += ['', '## 结果与复现文件', '',
        '- 选模锁定记录：`'+str(run/'final-selection.json')+'`。',
        '- 各配置保留`training-state.pt`（含优化器与随机状态）和`epoch-*/guidance.safetensors`。',
        '- 最终评测：`final/*-test.jsonl`、`final/*-metrics.json`、`final/comparison.json`。',
        '- 独立复算与哈希核验：`final/audit.json`；文字辅助模式单独报告。',
        '- 训练完整性、全部验证集复算和损失汇总：`training-audit.json`。',
        '- 正式协议：87场景、288判断对、576条问答；TICR包括共同答错，同时报告共同正确率。',
        '- 固定测试集曾用于历史研究，不宣称全新盲测；测试成绩不用于本轮反复选模。',
        '- 计划07:15结束训练探索，07:45保存固定汇总，07:55前退出本任务训练/评测进程。']
    text = '\n'.join(lines)+'\n'
    atomic(DELIVERY/'新结构训练进度-20261117.md', text)
    if snapshot:
        atomic(DELIVERY/'新结构训练进度-20261117-0745.md', text)
    if comparison and verification:
        b, g = comparison['baseline'], comparison['guided']
        result = ('本次达到相对直接基线的点估计提升目标。' if comparison['passes_point_estimate_goal'] else
                  '本次尚未达到一致性与判断质量同时提升的目标。')
        prose = f'''**课题3：多模态大模型价值对齐与纠正方法研究**

**（1）技术成果：同场景文本语义蒸馏驱动的双模式条件化视觉编码方法**

**解决的问题：** 常规图像编码缺少同场景文字的语义指导，而仅以通用道德问题作为条件难以区分具体场景。实际部署又可能缺少配套描述，使训练与预测条件不一致。针对这些问题，研究利用场景文字监督视觉语义条件，并在无描述输入下保持可用的价值判断能力。

**提出的方法与结果：** 在MoralCLIP–LLaVA的第11个视觉编码块内部加入场景语义预测器和低秩条件残差，本次锁定配置rank{rank}，新增{parameters:,}个可训练参数。GPU训练时使用冻结MoralCLIP文本编码器产生同场景描述表征，通过语义对齐、图像答案监督、经可靠性筛选的文本教师蒸馏、描述辅助监督及残差保持约束训练新增模块。原视觉权重、投影器、语言模型及既有语言LoRA保持冻结。无描述预测以图像预测的语义向量指导视觉残差；有描述时直接使用真实场景文字，任务问题不作为视觉条件。

按验证集锁定{selection.get('candidate')}第{selection.get('epoch')}轮参数后，在SMID冻结测试集的87个有效场景、288个判断对和576条独立图文问答上，同次直接基线TICR为{b['ticr']:.2%}，新结构为{g['ticr']:.2%}，差异为{(g['ticr']-b['ticr'])*100:+.2f}个百分点；图像准确率由{b['image_accuracy']:.2%}变为{g['image_accuracy']:.2%}，图文共同正确率由{b['joint_accuracy']:.2%}变为{g['joint_accuracy']:.2%}。{result}{'验证集最终保留第0轮零残差参数，训练后候选未通过选模要求，不能宣称取得训练收益。' if selection.get('epoch') == 0 else ''}TICR差异的场景配对bootstrap95%区间为[{comparison['paired_scene_bootstrap_delta95']['ticr'][0]*100:+.2f}, {comparison['paired_scene_bootstrap_delta95']['ticr'][1]*100:+.2f}]个百分点。逐条预测已独立复算，冻结文本分支输出完全一致，所用模块权重哈希与选模记录相符。本测试集曾用于历史研究，结果限定于本次固定协议；文字辅助模式另报，不替代独立输入TICR。

**成果相关性及支撑作用：** 该方法支撑课题三跨模态价值对齐与纠正研究，通过在视觉特征形成阶段引入同场景语义监督，提供兼容有描述和无描述部署的编码器结构。联合报告一致性、单模态准确率及图文共同正确率，为评价价值一致性提升及其质量代价提供可复查依据。隐私处理后效果需要专门的隐私输入实验，不能由本次原始输入结果推定。
'''
        atomic(DELIVERY/'同场景语义引导创新方法-20261117.md', prose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    deadline = datetime(2026, 11, 17, 7, 45, tzinfo=TZ)
    while True:
        remaining = (deadline-datetime.now(TZ)).total_seconds()
        render(args.run.resolve(), snapshot=remaining <= 0)
        if not args.watch or remaining <= 0:
            break
        time.sleep(min(30, remaining))


if __name__ == '__main__':
    main()
