"""Recompute every final benchmark from archived raw answers; CPU only."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moral_llava import polarity
from scripts.score_retrieval import verify as verify_retrieval
from scripts.moralise import score_moralise


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional new JSON report, outside archived results')
    args=parser.parse_args()
    data=ROOT/'data/polarity-v1'
    if not data.exists():
        from run_moral_llava import main as entry
        entry(['prepare'])
    rows=polarity.read(data/'test.json')
    audit=polarity.read(data/'audit.json')
    cases={}
    for name in ('majority','projector_only','trained'):
        cases['baseline/'+name]=(ROOT/f'results/baseline/{name}.jsonl',ROOT/f'results/baseline/{name}-metrics.json')
    for name in ('baseline','guided'):
        cases['text-guided-vision/'+name]=(ROOT/f'results/text-guided-vision/{name}-test.jsonl',ROOT/f'results/text-guided-vision/{name}-metrics.json')
    cases['text-guided-vision/final-trained']=(ROOT/'results/text-guided-vision/final-trained-test.jsonl',ROOT/'results/text-guided-vision/final-trained-metrics.json')
    for name in ('original-tower','qwen3-vl-8b','vlguard-posthoc','vlguard-mixed'):
        cases['benchmarks/'+name]=(ROOT/f'results/benchmarks/{name}/ticr.jsonl',ROOT/f'results/benchmarks/{name}/ticr-metrics.json')
    summary={}
    for name,(predictions,metrics) in cases.items():
        preds=read_jsonl(predictions)
        if any(p.get('data_sha256')!=polarity.sha(data/'test.json') for p in preds):
            raise ValueError('Frozen dataset SHA-256 changed: '+name)
        actual=polarity.score(rows,preds,audit)
        expected=polarity.read(metrics)
        expected=expected.get('metrics',expected)
        if not actual['complete_grouped_benchmark']:
            raise ValueError('Incomplete archived result: '+name)
        for group in ('overall','A','B'):
            a=actual['overall'] if group=='overall' else actual['groups'][group]
            b=expected['overall'] if group=='overall' else expected['groups'][group]
            if a != b:
                raise ValueError('Archived metric mismatch: '+name+'/'+group)
        summary[name]=actual['overall']['primary_macro']
    native_data=polarity.read(ROOT/'results/moralise/evaluation-data.json')
    native_predictions=read_jsonl(ROOT/'results/moralise/moralise.jsonl')
    native=score_moralise(native_data,native_predictions)
    original=polarity.read(ROOT/'results/moralise/moralise-metrics.json')
    if native != original:
        raise ValueError('MORALISE metrics changed')
    report={'status':'PASS','ticr':summary,'moralclip_r1':verify_retrieval(ROOT/'results/moralclip-r1/test.json'),
            'moralise':native,'raw_answer_count':sum(len(read_jsonl(p)) for p,_ in cases.values()),
            'moralise_answer_count':len(native_predictions)}
    if args.output:
        polarity.write_new(args.output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
