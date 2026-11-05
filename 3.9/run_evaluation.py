"""Portable evaluation of already generated private pairs; no data generation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Original SMID source_root')
    parser.add_argument('--private-data', type=Path, required=True, help='Frozen private pairs source_root')
    parser.add_argument('--ckpt', type=Path, required=True, help='Local MoralCLIP checkpoint directory')
    parser.add_argument('--output', type=Path, required=True, help='New result directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    args = parser.parse_args()
    for name in ('data', 'private_data', 'ckpt'):
        if not getattr(args, name).is_dir():
            parser.error(f'{name} must be an existing directory')
    if args.batch_size < 1:
        parser.error('batch size must be positive')
    archive = Path(__file__).resolve().parent/'moralclip_3.9_eps20_20261103.zip'
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    state = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
        'benchmark': 'SMID private-pair moral retrieval retention', 'workers': 1,
        'batch_size': args.batch_size, 'split': 'test', 'device': args.device,
        'data': str(args.data.resolve()), 'private_data': str(args.private_data.resolve()),
        'ckpt': str(args.ckpt.resolve()), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'generation': None, 'generation_note': 'Embedding retrieval; temperature/thinking do not apply',
        'intern_s2_status': 'not_evaluated'}
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='moral-privacy-eval-') as tmp:
            tmp = Path(tmp)
            with zipfile.ZipFile(archive) as package:
                for member in package.infolist():
                    target = (tmp/member.filename).resolve()
                    if not target.is_relative_to(tmp):
                        raise ValueError('Unsafe archive path')
                package.extractall(tmp)
            code = tmp/'3.9/moralclip_eval'
            config = json.loads((code/'privacy_pair_config.json').read_text())
            config.update(dataset_root=str(args.data.resolve()), model_dir=str(args.ckpt.resolve()),
                          output_dir=str(args.private_data.resolve()))
            config['evaluation'].update(output_dir=str(output/'results'), batch_size=args.batch_size,
                                        device=args.device, splits=['test'])
            config_path = output/'evaluation-config.json'
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
            command = [sys.executable, str(code/'evaluate_private_pairs.py'), '--config', str(config_path),
                       '--private-root', str(args.private_data.resolve()), '--splits', 'test']
            state['command'] = command
            with (output/'evaluation.log').open('w') as log:
                subprocess.run(command, cwd=code, check=True, stdout=log, stderr=subprocess.STDOUT)
            state['status'] = 'complete'
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        state.update(finished_at=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.monotonic()-started)
        (output/'run.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
