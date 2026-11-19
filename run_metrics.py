"""Local, offline entry for archived scoring and full model evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
CLIP_SHA = '4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a'
BASELINE_SHA = '318b609354f115162f14de9136005ded3cde31865dda8823559b1e0b79f79e2c'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'archive', 'baseline', 'scene', 'retrieval', 'all',
                                          'privacy-archive', 'privacy'))
    parser.add_argument('--output', type=Path, help='New output directory; default results/runs/<command>-<Beijing time>')
    parser.add_argument('--base-model', type=Path, default=ROOT/'models/base/llava')
    parser.add_argument('--moralclip-model', type=Path, default=ROOT/'models/base/moralclip')
    parser.add_argument('--privacy-project', type=Path,
                        default=ROOT.parent.parent/'privacy_polarity_20261115',
                        help='Existing privacy experiment folder; required for privacy commands')
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    output = (args.output or ROOT/'results/runs'/f'{args.command}-{stamp}').expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = str(args.base_model.expanduser().resolve())
    clip = str(args.moralclip_model.expanduser().resolve())
    privacy = args.privacy_project.expanduser().resolve()
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if not env.get(name, '').isdigit() or int(env[name]) < 1:
            env[name] = '8'
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    record = {'command': args.command, 'python': sys.executable, 'project': str(ROOT),
              'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'output': str(output), 'status': 'running', 'steps': []}

    def save():
        (output/'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')

    def run(name, *arguments):
        argv = [sys.executable, *map(str, arguments)]
        step = {'name': name, 'argv': argv, 'log': name+'.log'}
        record['steps'].append(step)
        save()
        print(f'运行 {name}；日志：{output / step["log"]}', flush=True)
        with (output/step['log']).open('x') as log:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        step['returncode'] = result.returncode
        save()
        if result.returncode:
            print((output/step['log']).read_text()[-5000:], file=sys.stderr)
            raise RuntimeError(f'{name} failed; see {output / step["log"]}')

    def metrics(relative):
        data = json.loads((output/relative).read_text())
        return data.get('metrics', data)['overall']['primary_macro']

    summary = {}
    try:
        save()
        if not (ROOT/'data/polarity-v1').exists():
            run('prepare', 'run_moral_llava.py', 'prepare')
        if args.command in ('check', 'archive'):
            run('repository', 'scripts/verify_repository.py')
        if args.command == 'check':
            run('verify-llava', 'scripts/download_models.py', 'llava', '--verify-local', base)
            run('verify-moralclip', 'scripts/download_models.py', 'moralclip', '--verify-local', clip)
            run('preflight', 'run_scene_guided_vision.py', 'preflight', '--base-model', base,
                '--moralclip-model', clip, '--expected-baseline-sha256', BASELINE_SHA,
                '--output', output/'preflight')
        if args.command == 'archive':
            run('archive', 'scripts/verify_results.py', '--output', output/'archive-verification.json')
            summary = json.loads((output/'archive-verification.json').read_text())
        if args.command == 'baseline':
            run('baseline-infer', 'run_moral_llava.py', 'infer', '--base-model', base,
                '--moralclip-model', clip, '--adapter', ROOT/'models/moralclip-llava-polarity',
                '--local-files-only', '--device', 'cuda', '--precision', 'bf16', '--batch-size', 1,
                '--output', output/'baseline.jsonl')
            run('baseline-score', 'run_moral_llava.py', 'score', '--predictions', output/'baseline.jsonl',
                '--audit', ROOT/'data/polarity-v1/audit.json', '--output', output/'baseline-metrics.json')
            summary['baseline'] = metrics('baseline-metrics.json')
        if args.command in ('privacy-archive', 'privacy'):
            predictions = privacy/'runs_fused/eval-private-test/predictions.jsonl'
            if args.command == 'privacy':
                predictions = output/'privacy.jsonl'
                run('privacy-infer', 'run_moral_llava.py', 'infer',
                    '--data', privacy/'polarity_fused/test.json',
                    '--image-root', privacy/'private/private_fused',
                    '--adapter', privacy/'runs_fused/adapter', '--base-model', base,
                    '--moralclip-model', clip, '--local-files-only', '--device', 'cuda',
                    '--precision', 'bf16', '--batch-size', 8, '--output', predictions)
            run('privacy-score', 'run_moral_llava.py', 'score',
                '--data', privacy/'polarity_fused/test.json', '--predictions', predictions,
                '--audit', privacy/'polarity_fused/audit.json', '--output', output/'privacy-metrics.json')
            summary['privacy-fused'] = metrics('privacy-metrics.json')
            if args.command == 'privacy-archive':
                actual = json.loads((output/'privacy-metrics.json').read_text())
                expected = json.loads((privacy/'runs_fused/eval-private-test/metrics.json').read_text())
                if actual != expected:
                    raise ValueError('Recomputed privacy metrics differ from the archive')
        if args.command in ('scene', 'all'):
            run('scene-infer', 'run_scene_guided_vision.py', 'infer', '--base-model', base,
                '--moralclip-model', clip, '--baseline-adapter', ROOT/'models/moralclip-llava-polarity',
                '--expected-baseline-sha256', BASELINE_SHA, '--guidance', ROOT/'models/scene-guided-vision',
                '--precision', 'bf16', '--eval-batch-size', 1, '--evaluate-paired', '--output', output/'scene')
            for name in ('baseline', 'guided', 'paired-assisted'):
                summary[name] = metrics(f'scene/{name}-metrics.json')
        if args.command in ('retrieval', 'all'):
            run('retrieval', 'run_retrieval.py', '--model', clip, '--expected-model-sha256', CLIP_SHA,
                '--device', 'cuda', '--output', output/'retrieval/test.json')
            run('retrieval-score', 'scripts/score_retrieval.py', '--evaluation', output/'retrieval/test.json')
            retrieved = json.loads((output/'retrieval/test.json').read_text())
            summary['retrieval'] = {direction: {key: retrieved['directions'][direction][key]
                for key in ('r_at_1', 'correct_queries', 'num_queries')}
                for direction in ('image_to_text', 'text_to_image')}
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        record.update(status='complete', finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        save()
        print('完成，结果目录：'+str(output))
        if args.command in ('baseline', 'scene', 'all', 'privacy', 'privacy-archive'):
            for name, data in summary.items():
                if 'ticr' in data:
                    print(f"{name}: TICR={data['ticr']:.2%}, 图像准确率={data['image_accuracy']:.2%}, "
                          f"文本准确率={data['text_accuracy']:.2%}, 共同正确率={data['joint_accuracy']:.2%}")
    except Exception as error:
        record.update(status='failed', error=str(error))
        save()
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
