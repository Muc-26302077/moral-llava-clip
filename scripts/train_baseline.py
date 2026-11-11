"""Train the final two-stage MoralCLIP–LLaVA baseline and evaluate its answers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New run directory')
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--moralclip-model', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Two updates per stage, train/validation only')
    args = parser.parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error('Batch size and gradient accumulation must be positive')
    from moral_llava import polarity
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Baseline training requires a CUDA GPU')
    data = ROOT/'data/polarity-v1'
    if not data.exists():
        from run_moral_llava import main as entry
        entry(['prepare'])
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    derived = output/'data'
    derived.mkdir()
    for split in ('train', 'validation'):
        rows = polarity.read(data/f'{split}.json')
        if args.smoke:
            scenes = {r['scene_id'] for r in rows[:48]}
            rows = [r for r in rows if r['scene_id'] in scenes]
        polarity.write_new(derived/f'{split}.json', rows)
        polarity.write_new(derived/f'{split}-images.json', [r for r in rows if r['modality']=='image'])
    common = ['--base-model', str(args.base_model.expanduser().resolve()),
              '--moralclip-model', str(args.moralclip_model.expanduser().resolve()),
              '--local-files-only', '--batch-size', str(args.batch_size),
              '--eval-batch-size', str(args.batch_size), '--precision', 'bf16',
              '--gradient-accumulation-steps', str(args.gradient_accumulation),
              '--seed', '42', '--logging-steps', '25',
              '--gradient-checkpointing' if args.gradient_checkpointing else '--no-gradient-checkpointing']
    if args.smoke:
        common += ['--max-steps', '2', '--logging-steps', '1']
    commands = [
        ['train'] + common + ['--moralclip-stage', 'projector', '--epochs', '1',
            '--learning-rate', '1e-3', '--projector-learning-rate', '1e-3',
            '--train-data', str(derived/'train-images.json'),
            '--validation-data', str(derived/'validation-images.json'),
            '--output-dir', str(output/'projector')],
        ['train'] + common + ['--moralclip-stage', 'joint', '--epochs', '3',
            '--learning-rate', '2e-5', '--projector-learning-rate', '1e-4',
            '--initial-adapter', str(output/'projector'),
            '--train-data', str(derived/'train.json'),
            '--validation-data', str(derived/'validation.json'),
            '--output-dir', str(output/'adapter')],
    ]
    if args.smoke:
        commands += [['infer', '--adapter', str(output/'adapter'),
            '--base-model', str(args.base_model.resolve()),
            '--moralclip-model', str(args.moralclip_model.resolve()), '--local-files-only',
            '--data', str(derived/'validation.json'), '--limit', '8', '--device', 'cuda',
            '--precision', 'bf16', '--output', str(output/'smoke-predictions.jsonl')]]
    else:
        for name, adapter in (('projector_only', output/'projector'), ('trained', output/'adapter')):
            commands += [
                ['infer', '--adapter', str(adapter), '--base-model', str(args.base_model.resolve()),
                 '--moralclip-model', str(args.moralclip_model.resolve()), '--local-files-only',
                 '--device', 'cuda', '--precision', 'bf16', '--batch-size', '8',
                 '--output', str(output/f'{name}.jsonl')],
                ['score', '--predictions', str(output/f'{name}.jsonl'),
                 '--output', str(output/f'{name}-metrics.json')],
            ]
    (output/'commands.json').write_text(json.dumps(commands, ensure_ascii=False, indent=2)+'\n')
    for index, command in enumerate(commands):
        print(f'[{index+1}/{len(commands)}] {command[0]}', flush=True)
        with (output/f'step-{index+1}.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(ROOT/'run_moral_llava.py')] + command,
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed:', output)


if __name__ == '__main__':
    main()
