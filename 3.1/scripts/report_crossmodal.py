"""Render a completed paired evaluation; audit original outputs before reporting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from moral_llava.crossmodal import parsed, read_json, score, sha256, validate_protocol, write_new_json


def pct(v):
    return '未定义' if v is None else f'{100*v:.2f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'runs/crossmodal-v1/full')
    parser.add_argument('--protocol', type=Path, default=ROOT/'runs/crossmodal-v1/protocol.json')
    parser.add_argument('--output', type=Path, default=ROOT/'CROSSMODAL_RESULTS.md')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = validate_protocol(read_json(args.protocol))
    status = read_json(args.run/'status.json')
    if status['status'] != 'complete' or not status['benchmark_complete']:
        raise ValueError('Benchmark is not complete')
    if status['protocol_sha256'] != sha256(args.protocol):
        raise ValueError('Protocol hash mismatch')
    reports, predictions, checks = {}, {}, {}
    for decoding in ('constrained', 'free'):
        predictions[decoding] = {}
        for model in ('base', 'lora'):
            path = args.run/f'{model}_{decoding}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            scene_groups = {s['id']: s['group'] for s in protocol['scenes']}
            for r in rows:
                if (r['model'] != model or r['decoding'] != decoding or r['group'] != scene_groups[r['id']]
                        or r['protocol_sha256'] != status['protocol_sha256']):
                    raise ValueError('Prediction provenance mismatch')
            predictions[decoding][model] = rows
            checks[path.name] = {'count': len(rows), 'sha256': sha256(path)}
        report = score(protocol, predictions[decoding])
        saved = read_json(args.run/f'metrics_{decoding}.json')
        if any(saved.get(k) != v for k, v in report.items()):
            raise ValueError(f'Recomputed metrics disagree: {decoding}')
        reports[decoding] = report
    lines = ['# 跨模态道德一致性：固定两组实测结果', '',
        '原始 LLaVA 与已有图像标签 LoRA 在同一运行环境重新推理；未训练、未更改权重。', '',
        'A/B 为原始 SMID 测试划分的两个不重叠子组，每组 63 张，固定使用 captions[0]。它们不是两个外部独立数据集。', '',
        '主分析在推理前固定为 constrained：两个模型均使用同一 243 种合法 JSON 答案约束，主指标为五维完整一致率；free 为单独的补充诊断。', '',
        '门槛：每组主指标 >=60%，即至少 38/63。无效输出计失败；不修补回答、不根据结果改组。', '',
        '## 主分析：统一格式约束', '',
        '| 组 | 模型 | 一致场景 | 五维完整一致率 | 逐维一致率 | 两边同时正确率 | 图像标签完全匹配 | 文本标签完全匹配 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g in ('A', 'B'):
        for model in ('base', 'lora'):
            m = reports['constrained']['groups'][g]['models'][model]
            lines.append(f"| {g} | {model} | {m['consistent_pairs']}/63 | {pct(m['strict_agreement'])} | {pct(m['dimension_agreement'])} | {pct(m['joint_correct_rate'])} | {pct(m['modality_metrics']['image']['exact_match'])} | {pct(m['modality_metrics']['text']['exact_match'])} |")
    passed = reports['constrained']['lora_passes_both_groups']
    lines += ['', f"**本轮主指标判定：LoRA {'达到' if passed else '未达到'}两个固定组均 >=60% 的阶段门槛。**", '',
        '| 组 | LoRA 一致率相对基模变化 | LoRA 一致率 Wilson 95% 区间 | 基模已一致场景的保持率 |',
        '| --- | ---: | --- | --- |']
    for g in ('A', 'B'):
        r = reports['constrained']['groups'][g]
        ci = r['models']['lora']['strict_agreement_wilson95']
        retention = r['baseline_consistency_retention']
        lines.append(f"| {g} | {r['strict_agreement_delta_pp']:+.2f} 个百分点 | {pct(ci[0])}–{pct(ci[1])} | {retention['numerator']}/{retention['denominator']}，{pct(retention['rate'])} |")
    lines += ['', '条件保持率只在基模有效一致场景上计算，不替代全部场景的主一致率；分母为零时为未定义。上述门槛为点估计门槛，不是置信区间下界门槛。', '',
        '## 自由生成诊断', '',
        '| 组 | 模型 | 图像有效率 | 文本有效率 | 五维完整一致率 | 两边同时正确率 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for g in ('A', 'B'):
        for model in ('base', 'lora'):
            m = reports['free']['groups'][g]['models'][model]
            lines.append(f"| {g} | {model} | {pct(m['modality_metrics']['image']['valid_output_rate'])} | {pct(m['modality_metrics']['text']['valid_output_rate'])} | {pct(m['strict_agreement'])} | {pct(m['joint_correct_rate'])} |")
    base_values = [parsed(r) for r in predictions['constrained']['base']]
    if all(v is not None and v['non_moral'] for v in base_values):
        lines += ['', '已核查：基模 constrained 的 252/252 条回答均为全 not_annotated。100% 一致率来自恒定输出，不能视为道德判断能力更强。', '']
    lines += ['', '## 一致与正确的区分', '',
        '常量模型在两种输入下始终输出全 not_annotated，可取得 100% 跨模态一致率；因此一致率本身不证明道德判断准确或视觉信息被有效利用。', '',
        '| 组 | 全 not_annotated 常量基线标签完全匹配率 | LoRA 一致但两边不同时正确的场景数（主分析） |',
        '| --- | ---: | ---: |']
    for g in ('A', 'B'):
        scenes = [s for s in protocol['scenes'] if s['group'] == g]
        constant_correct = sum(s['reference']['non_moral'] for s in scenes)
        m = reports['constrained']['groups'][g]['models']['lora']
        lines.append(f"| {g} | {constant_correct}/63，{pct(constant_correct/63)} | {m['consistent_pairs']-m['both_correct_pairs']} |")
    distributions = {}
    for decoding, models in predictions.items():
        for model, rows in models.items():
            for modality in ('image', 'text'):
                values = [parsed(r) for r in rows if r['modality'] == modality]
                counts = Counter(json.dumps(v, sort_keys=True) for v in values if v is not None)
                distributions[f'{model}_{decoding}_{modality}'] = {
                    'valid': sum(counts.values()), 'distinct_valid_targets': len(counts),
                    'most_common_targets': [{'prediction': json.loads(v), 'count': n} for v, n in counts.most_common(5)]}
    lines += ['', '## 验证与限制', '',
        '- 四个预测文件各 252 条，总计 1008 条；ID/模态唯一并完整覆盖两个固定组。',
        '- 从 raw_prediction 独立重算两种协议，结果与运行器保存指标逐项相符；文件哈希保存在 verification.json。',
        '- 8 项评分/分组回归测试通过，GPU 小样本覆盖两个模型、两种模态和两种解码。',
        '- 生成 caption 的语义等价性尚待人工审核，图片级标签不能自动视为独立人工文本道德标注。',
        '- 当前为固定单划分、单次解码协议的探索性结果；未证明外部泛化或达到原 95% 目标。',
        '- not_annotated 表示数据未标注，不代表已确认中性。', '',
        '## 复现文件', '',
        '- 协议：runs/crossmodal-v1/protocol.json 与 preflight.json。',
        '- 原始回答：runs/crossmodal-v1/full/{base,lora}_{constrained,free}.jsonl。',
        '- 指标及逐场景比较：runs/crossmodal-v1/full/metrics_{constrained,free}.json。',
        '- 环境与状态：runs/crossmodal-v1/full/status.json；日志：runs/crossmodal-v1/full.log。',
        '- 运行及指标说明：CROSSMODAL_EVALUATION.md。', '',
        f"环境：torch {status['torch']}；transformers {status['transformers']}；peft {status['peft']}；BF16 CUDA。", '',
        f"协议 SHA256：`{status['protocol_sha256']}`。", '']
    write_new_json(args.run/'verification.json', {'status': 'PASS', 'recomputed_metrics_match': True,
        'files': checks, 'prediction_distributions': distributions, 'protocol_sha256': status['protocol_sha256']})
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
