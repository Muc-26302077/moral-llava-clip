"""One-command frozen polarity evaluation: data + adapter checkpoint -> answers + TICR."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from delivery import ROOT, save, sha, validate
sys.path.insert(0, str(ROOT))
from moral_llava import polarity


def load_delivery(root):
    validate(root)
    rows = []
    for path in sorted(Path(root).rglob('*.jsonl')):
        if any(x.startswith('.') for x in path.relative_to(root).parts):
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            obj = obj.get('trace', obj)
            row = {k: v for k, v in obj.items() if k != 'messages'}
            turns = []
            for m in obj['messages']:
                if m['role'] not in ('user', 'assistant'):
                    raise ValueError('Frozen polarity evaluation requires user/assistant turns')
                content = m['content']
                if isinstance(content, list):
                    text = []
                    for b in content:
                        if b['type'] == 'image':
                            if 'image' in row:
                                raise ValueError('Frozen polarity evaluation permits one image')
                            row['image'] = b['image']
                            text.append('<image>\n')
                        else:
                            text.append(b['text'])
                    content = ''.join(text)
                turns.append({'from': 'human' if m['role'] == 'user' else 'gpt', 'value': content})
            row['conversations'] = turns
            rows.append(row)
    return rows


def verify_frozen(rows, reference):
    if len(rows) != len(reference):
        raise ValueError('Evaluation must cover the full frozen test set')
    fields = ('id', 'scene_id', 'split', 'protocol', 'task', 'modality', 'group', 'caption_index', 'caption', 'conversations')
    index = {r['id']: r for r in reference}
    seen = set()
    for r in rows:
        if r['id'] in seen or r['id'] not in index:
            raise ValueError('Unexpected or duplicate evaluation ID')
        seen.add(r['id'])
        ref = index[r['id']]
        if any(r.get(k) != ref.get(k) for k in fields):
            raise ValueError(f'Frozen prompt/target/metadata changed: {r["id"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Standard test source_root or frozen legacy test.json')
    p.add_argument('--ckpt', type=Path, required=True, help='MoralCLIP-LLaVA polarity adapter directory')
    p.add_argument('--output', type=Path, required=True, help='New result directory')
    p.add_argument('--image-root', type=Path, default=ROOT/'data/raw/SMID_moralclip')
    p.add_argument('--moralclip-model', default='anaaa2/moralclip-base')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu', 'mps'), default='cuda')
    p.add_argument('--precision', choices=('bf16', 'fp16', 'fp32'), default='bf16')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--predictions', type=Path, help='Replay archived answers; no new inference or timing claim')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch size must be positive')
    weight = args.ckpt/'adapter_model.safetensors'
    if not weight.is_file():
        p.error('ckpt requires actual adapter weights; run git lfs pull first')
    with weight.open('rb') as stream:
        if stream.read(43).startswith(b'version https://git-lfs.github.com/spec/v1'):
            p.error('ckpt requires actual adapter weights; run git lfs pull first')
    adapter_sha = sha(weight)
    data = args.data.resolve()
    image_root = data if data.is_dir() else args.image_root.resolve()
    rows = load_delivery(data) if data.is_dir() else json.loads(data.read_text())
    with tempfile.TemporaryDirectory(prefix='frozen-polarity-eval-') as tmp:
        tmp = Path(tmp)
        audit = polarity.prepare(ROOT/'data/raw/SMID_moralclip', tmp/'reference',
                    ROOT/'results/reference/runs/crossmodal-v1/protocol.json', ROOT/'data/excluded_images.json')
        reference = json.loads((tmp/'reference/test.json').read_text())
        verify_frozen(rows, reference)
        # Verify image identity as well as textual prompts before running a model.
        original = {r['id']: r for r in reference}
        for row in rows:
            if row['modality'] == 'image':
                if sha(image_root/row['image']) != sha(ROOT/'data/raw/SMID_moralclip'/original[row['id']]['image']):
                    raise ValueError(f'Test image differs from frozen bytes: {row["id"]}')
        args.output.mkdir(parents=True, exist_ok=False)
        infer_data = args.output/'evaluation-data.json'
        save(infer_data, rows)
        metadata = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
            'mode': 'archive-replay' if args.predictions else 'model-inference',
            'protocol': polarity.VERSION, 'records': len(rows), 'pairs': len(rows)//2,
            'workers': 1, 'batch_size': args.batch_size, 'ckpt': str(args.ckpt.resolve()),
            'adapter_sha256': adapter_sha,
            'generation': {'do_sample': False, 'temperature': None, 'max_new_tokens': 16,
                'max_length': 2048, 'thinking': False, 'precision': args.precision},
            'runtime_seconds': None, 'inference_seconds': None}
        save(args.output/'run.json', metadata)
        started = time.monotonic()
        try:
            predictions = args.output/'answers.jsonl'
            if args.predictions:
                predictions.write_bytes(args.predictions.read_bytes())
            else:
                command = [sys.executable, str(ROOT/'run_moralclip_polarity.py'), 'infer',
                    '--data', str(infer_data), '--image-root', str(image_root), '--adapter', str(args.ckpt.resolve()),
                    '--moralclip-model', args.moralclip_model, '--output', str(predictions),
                    '--batch-size', str(args.batch_size), '--device', args.device, '--precision', args.precision,
                    '--max-new-tokens', '16', '--max-length', '2048']
                if args.local_files_only:
                    command += ['--local-files-only']
                metadata['command'] = command
                before = time.monotonic()
                with (args.output/'inference.log').open('w') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                metadata['inference_seconds'] = time.monotonic()-before
            answers = [json.loads(x) for x in predictions.read_text().splitlines() if x.strip()]
            if any(a.get('adapter_sha256') != adapter_sha for a in answers):
                raise ValueError('Answers do not match the requested checkpoint')
            if not args.predictions and any(a.get('data_sha256') != sha(infer_data) for a in answers):
                raise ValueError('Answers do not match the evaluated data')
            metrics = polarity.score(rows, answers, audit)
            if not metrics['complete_grouped_benchmark']:
                raise ValueError('Frozen grouped benchmark incomplete')
            save(args.output/'metrics.json', metrics)
            metadata.update(status='complete', answer_sha256=sha(predictions), data_sha256=sha(infer_data))
        except Exception as error:
            metadata.update(status='failed', error=str(error))
            raise
        finally:
            metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.monotonic()-started)
            save(args.output/'run.json', metadata)
    print(json.dumps(metrics['overall']['primary_macro'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
