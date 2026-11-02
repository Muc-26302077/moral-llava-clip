"""Portable binary QA entry point; existing run.py remains unchanged."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('prepare','train','infer','score','majority'):
        print('Usage: run_binary.py {prepare,train,infer,score,majority} [options]')
        return 0 if argv in ([],['--help'],['-h']) else 2
    command, extra = argv[0], argv[1:]
    raw, data = ROOT/'data/raw/SMID_moralclip', ROOT/'data/binary-v1'
    frozen = ROOT/'results/reference/runs/crossmodal-v1/protocol.json'
    if not frozen.exists():
        frozen = ROOT/'runs/crossmodal-v1/protocol.json'
    adapter = ROOT/'models/image-lora'
    if not adapter.exists():
        adapter = ROOT/'runs/moral-llava-1.5'
    if command == 'train':
        from moral_llava.train import main as train
        defaults = ['--train-data',str(data/'train.json'),'--validation-data',str(data/'validation.json'),
                    '--image-root',str(raw),'--output-dir',str(ROOT/'results/runs/binary-sft-v1'),
                    '--revision','b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
                    '--epochs','3','--batch-size','4','--eval-batch-size','4',
                    '--gradient-accumulation-steps','2','--learning-rate','2e-5',
                    '--seed','42','--precision','bf16']
        if adapter.exists():
            defaults += ['--initial-adapter',str(adapter)]
        return train(defaults+extra+['--task','binary'])
    if command == 'infer':
        from moral_llava.infer import main as infer
        defaults = ['--data',str(data/'test.json'),'--image-root',str(raw),
                    '--output',str(ROOT/'results/runs/binary-predictions.jsonl'),
                    '--max-new-tokens','8']
        if adapter.exists():
            defaults += ['--adapter',str(adapter)]
        return infer(defaults+extra+['--task','binary'])
    from moral_llava.binary import main as binary
    if command == 'prepare':
        defaults = ['--image-root',str(raw),'--output',str(data)]
        if frozen.exists():
            defaults += ['--groups-protocol',str(frozen)]
        excluded = ROOT/'data/excluded_images.json'
        if excluded.exists():
            defaults += ['--excluded-images',str(excluded)]
    else:
        defaults = ['--data',str(data/'test.json')]
        if command == 'majority':
            defaults += ['--train-data',str(data/'train.json')]
    return binary([command]+defaults+extra) or 0


if __name__ == '__main__':
    raise SystemExit(main())
