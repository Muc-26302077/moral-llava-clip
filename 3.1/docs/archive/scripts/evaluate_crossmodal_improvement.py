"""Compare a validation-selected adapter with the frozen old LoRA test run."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from moral_llava.crossmodal import read_json,sha256,score,parsed,write_new_json


def pct(v):return f'{100*v:.2f}%'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-run',type=Path,required=True)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--old-run',type=Path,default=ROOT/'runs/crossmodal-v1/full')
    parser.add_argument('--protocol',type=Path,default=ROOT/'runs/crossmodal-v1/protocol.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    protocol=read_json(args.protocol);protocol_hash=sha256(args.protocol)
    status=read_json(args.new_run/'status.json');selection=read_json(args.training_run/'selection.json')
    if status['status']!='complete' or status['protocol_sha256']!=protocol_hash:
        raise ValueError('Incomplete or mismatched run')
    if Path(status['arguments']['adapter']).resolve()!=Path(selection['selected_adapter']).resolve():
        raise ValueError('Adapter does not match validation selection')
    args.output.mkdir(parents=True,exist_ok=False)
    records={};reports={};pair_changes={}
    def load(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    for decoding in ('constrained','free'):
        original=load(args.old_run/f'lora_{decoding}.jsonl')
        candidate=load(args.new_run/f'lora_{decoding}.jsonl')
        base=load(args.old_run/f'base_{decoding}.jsonl')
        old_index={(r['id'],r['modality']):r for r in original}
        for r in candidate:
            prior=old_index[r['id'],r['modality']]
            if r['protocol_sha256']!=protocol_hash or r['prompt_sha256']!=prior['prompt_sha256'] or r['decoding']!=decoding:
                raise ValueError('Prompt/protocol/decoding changed')
        report=score(protocol,{'base':base,'lora':candidate})
        write_new_json(args.output/f'metrics_{decoding}.json',report)
        old_report=read_json(args.old_run/f'metrics_{decoding}.json')
        reports[decoding]={'old':old_report,'new':report}
        records[decoding]={'old':original,'new':candidate}
        new_index={(r['id'],r['modality']):r for r in candidate}
        changes={}
        for g in ('A','B'):
            old_set,new_set=set(),set()
            for scene in protocol['scenes']:
                if scene['group']!=g:continue
                identifier=scene['id']
                for index,result in [(old_index,old_set),(new_index,new_set)]:
                    a,b=[parsed(index[identifier,m]) for m in ('image','text')]
                    if a is not None and b is not None and a==b:result.add(identifier)
            changes[g]={'old_consistent':len(old_set),'new_consistent':len(new_set),
                        'gained':sorted(new_set-old_set),'lost':sorted(old_set-new_set),
                        'retained':len(new_set&old_set),'retention_rate':len(new_set&old_set)/len(old_set) if old_set else None}
        pair_changes[decoding]=changes
    main_report=reports['constrained']['new']
    lines=['# 图文联合 LoRA：一致性改进实测','','选中 checkpoint：`'+selection['selected_adapter']+'`。',
           '模型只按训练/验证数据优化和选择；测试复用先前固定的 126 场景 A/B 分组、caption、问题及解码设置。',
           '由于本测试划分此前已查看，本轮属于已知划分上的方法改进，不是新的独立盲测。','',
           '## 主分析：五维完整一致率','','| 组 | 模型 | 一致场景 | 完整一致率 | 逐维一致率 | 两边都正确 | 图像标签完全匹配 | 文本标签完全匹配 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g in ('A','B'):
        for version,label in [('old','旧图像 LoRA'),('new','图文联合 LoRA')]:
            m=reports['constrained'][version]['groups'][g]['models']['lora']
            lines.append(f"| {g} | {label} | {m['consistent_pairs']}/63 | {pct(m['strict_agreement'])} | {pct(m['dimension_agreement'])} | {pct(m['joint_correct_rate'])} | {pct(m['modality_metrics']['image']['exact_match'])} | {pct(m['modality_metrics']['text']['exact_match'])} |")
    passed=main_report['lora_passes_both_groups']
    lines+=['',f"**结果：新模型{'达到' if passed else '未达到'}两个固定组完整一致率均 >=60% 的阶段要求。**",'',
            '## 对旧模型一致场景的保持与变化','','| 组 | 保留旧一致场景 | 新增一致场景 | 失去一致场景 |',
            '| --- | ---: | ---: | ---: |']
    for g in ('A','B'):
        c=pair_changes['constrained'][g]
        lines.append(f"| {g} | {c['retained']}/{c['old_consistent']} | {len(c['gained'])} | {len(c['lost'])} |")
    lines+=['','## 自由生成补充结果','','| 组 | 新模型图像有效率 | 新模型文本有效率 | 五维完整一致率 |','| --- | ---: | ---: | ---: |']
    for g in ('A','B'):
        m=reports['free']['new']['groups'][g]['models']['lora']
        lines.append(f"| {g} | {pct(m['modality_metrics']['image']['valid_output_rate'])} | {pct(m['modality_metrics']['text']['valid_output_rate'])} | {pct(m['strict_agreement'])} |")
    lines+=['','## 方法与边界','',
        '- 从旧图像 LoRA 继续训练；视觉塔、原多模态投影器和基础语言权重冻结，仅更新语言 LoRA。',
        '- 每个训练场景的图片与描述监督为同一标签，同时加入五维状态 CE 及状态概率 JS 一致性损失。',
        '- JS 计算在 teacher forcing 下的五个状态首 token 上，最终指标通过两路独立生成测量。',
        '- 3 轮训练，每轮保存并验证；按验证集一致率及准确率约束选择 checkpoint。',
        '- 未改变测试分组、描述、问题、严格五维指标或 60% 门槛；新旧模型每条 prompt 哈希均核对相同。',
        '- 原始基模在格式约束下全 not_annotated 的退化问题仍存在；旧实验的基模预测完整保留。',
        '- 一致不等于正确，需结合两边都正确率和标签准确率；生成描述语义人工审核仍待补充。',
        '- 阈值基于点估计，Wilson 95% 区间见 JSON；不能将点估计达标解释为区间下界达标。','',
        '详见 CROSSMODAL_IMPROVEMENT_PLAN.md；逐条预测在新评测目录；本报告旁的 JSON 保留完整指标及场景变化。','']
    (args.output/'report.md').write_text('\n'.join(lines))
    distributions={}
    for decoding,versions in records.items():
        for version,rows in versions.items():
            for modality in ('image','text'):
                counts=Counter(json.dumps(parsed(r),sort_keys=True) for r in rows if r['modality']==modality)
                distributions[f'{version}_{decoding}_{modality}']=[{'prediction':json.loads(s),'count':n} for s,n in counts.most_common(5)]
    write_new_json(args.output/'verification.json',{'status':'PASS','prompts_identical':True,'protocol_sha256':protocol_hash,
        'training_selection':selection,'pair_changes':pair_changes,'dominant_predictions':distributions,
        'new_prediction_sha256':{p.name:sha256(p) for p in args.new_run.glob('*.jsonl')}})
    print(args.output/'report.md')


if __name__=='__main__':main()
