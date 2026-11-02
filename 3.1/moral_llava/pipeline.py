"""Portable entry point for the released MoralLLaVA evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .crossmodal import ROOT, REVISION, prepare, read_json, score, sha256, write_new_json

MODEL_ID = 'llava-hf/llava-1.5-7b-hf'
REFERENCE = ROOT / 'results/reference'
ADAPTER = ROOT / 'models/moralllava'


def output_directory(command, specified=None):
    destination = (Path(specified).expanduser().resolve() if specified else
                   ROOT/'results/runs'/f'{command}-{datetime.now():%Y%m%d-%H%M%S-%f}')
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def read_predictions(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def rescore(destination):
    protocol_path = REFERENCE/'runs/crossmodal-v1/protocol.json'
    protocol = read_json(protocol_path)
    for decoding in ('constrained', 'free'):
        base = read_predictions(REFERENCE/f'runs/crossmodal-v1/full/base_{decoding}.jsonl')
        lora = read_predictions(REFERENCE/f'runs/crossmodal-joint-v1-test/lora_{decoding}.jsonl')
        if any(row['protocol_sha256'] != sha256(protocol_path) for row in base+lora):
            raise ValueError('Saved predictions do not match the frozen protocol')
        metrics = score(protocol, {'base':base, 'lora':lora})
        write_new_json(destination/f'metrics_{decoding}.json', metrics)
    return metrics


def check_adapter():
    path = ADAPTER/'adapter_model.safetensors'
    if not path.is_file():
        raise FileNotFoundError('Missing LoRA weights. Run git lfs install and git lfs pull in the repository.')
    with path.open('rb') as stream:
        if stream.read(64).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('LoRA file is a Git LFS pointer. Run git lfs pull first.')
    expected = read_json(REFERENCE/'runs/crossmodal-joint-v1-test/status.json')['adapter_sha256']
    if sha256(path) != expected:
        raise ValueError('LoRA SHA-256 differs from the validation-selected model')
    if read_json(ADAPTER/'training_metadata.json')['resolved_base_commit'] != REVISION:
        raise ValueError('LoRA base revision mismatch')


def resolve_base(specified=None, offline=False):
    from .download_model import ALLOW_PATTERNS, EXPECTED_WEIGHTS, verify_weight

    if specified:
        snapshot = Path(specified).expanduser().resolve()
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError('Install the environment first: conda env create -f environment.yml') from error
        snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION,
                        allow_patterns=ALLOW_PATTERNS, local_files_only=offline)).resolve()
    if not snapshot.is_dir() or snapshot.name != REVISION:
        raise ValueError(f'--base-model must be a snapshot directory named {REVISION}')
    checks = [verify_weight(snapshot, name, expected) for name, expected in EXPECTED_WEIGHTS.items()]
    if not all(item['ok'] for item in checks):
        failed = ', '.join(item['file'] for item in checks if not item['ok'])
        raise ValueError(f'Pinned base weight verification failed: {failed}')
    return snapshot, checks


def evaluate(args):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('Install the environment first: conda env create -f environment.yml') from error
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Use score for existing predictions, or prepare on a download host.')
    if args.device == 'cuda' and args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support BF16; select a supported GPU or --precision fp32.')
    check_adapter()
    snapshot, checks = resolve_base(args.base_model, args.offline)
    destination = output_directory('evaluate', args.output)
    protocol = prepare(ROOT/'data/raw/SMID_moralclip/test/smid_manifest.json',
                       ROOT/'data/processed/test.json', ROOT/'data/raw/SMID_moralclip')
    protocol_path = destination/'protocol.json'
    write_new_json(protocol_path, protocol)
    write_new_json(destination/'base_model.json', {'model_id':MODEL_ID, 'revision':REVISION,
                   'snapshot_path':str(snapshot), 'files':checks})
    from .infer_crossmodal import run
    run(SimpleNamespace(protocol=protocol_path, base_model=snapshot, adapter=ADAPTER,
        output=destination/'predictions', device=args.device, precision=args.precision,
        batch_size=args.batch_size, max_new_tokens=256, cpu_threads=8,
        decoding=args.decoding, models='both', limit=args.limit))
    return destination


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('score','verify','prepare','evaluate','infer'))
    parser.add_argument('--output','--run-dir', dest='output', type=Path,
                        help='New output directory; relative paths are resolved from your working directory')
    parser.add_argument('--base-model', type=Path, default=os.environ.get('BASE_SNAPSHOT'),
                        help='Optional pinned local snapshot; otherwise download/reuse the Hugging Face cache')
    parser.add_argument('--offline', action='store_true', help='Use cached base weights without network access')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    parser.add_argument('--precision', choices=('bf16','fp32'), default='bf16')
    parser.add_argument('--decoding', choices=('constrained','free','both'), default='both')
    parser.add_argument('--limit', type=int, default=0, help='Smoke only: limit scenes and do not write full metrics')
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.limit < 0:
        parser.error('batch-size must be positive; limit must be nonnegative')
    if args.command == 'verify' and args.output:
        parser.error('verify does not use --output')
    try:
        if args.command == 'verify':
            from scripts.verify_release import verify
            report = verify()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == 'score':
            destination = output_directory('score', args.output)
            rescore(destination)
        elif args.command == 'prepare':
            check_adapter()
            snapshot, checks = resolve_base(args.base_model, args.offline)
            destination = output_directory('prepare', args.output)
            write_new_json(destination/'base_model.json', {'model_id':MODEL_ID,
                           'revision':REVISION, 'snapshot_path':str(snapshot), 'files':checks})
        else:
            destination = evaluate(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')
    print(f'Results: {destination}')
    return 0
