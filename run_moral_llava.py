"""Prepare SMID, train Moral-LLaVA, generate answers, and score TICR on CPU."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('prepare', 'train', 'infer', 'score', 'majority'):
        print('Usage: run_moral_llava.py {prepare,train,infer,score,majority} [options]')
        return 0 if argv in ([], ['-h'], ['--help']) else 2
    command, extra = argv[0], argv[1:]
    data = ROOT/'data/polarity-v1'
    raw = ROOT/'data/raw/SMID_moralclip'
    if command == 'train':
        from moral_llava.train import main as train
        return train(['--task', 'polarity', '--image-root', str(raw),
                      '--train-data', str(data/'train.json'),
                      '--validation-data', str(data/'validation.json'),
                      '--revision', 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'] + extra)
    if command == 'infer':
        from moral_llava.infer import main as infer
        return infer(['--task', 'polarity', '--image-root', str(raw),
                      '--data', str(data/'test.json'), '--max-new-tokens', '16',
                      '--adapter', str(ROOT/'models/moralclip-llava-polarity')] + extra)
    from moral_llava.polarity import main as polarity
    if command == 'prepare':
        defaults = ['--image-root', str(raw), '--output', str(data),
                    '--groups-protocol', str(ROOT/'data/groups.json'),
                    '--excluded-images', str(ROOT/'data/excluded_images.json')]
    else:
        defaults = ['--data', str(data/'test.json')]
        if command == 'majority':
            defaults += ['--train-data', str(data/'train.json')]
    return polarity([command] + defaults + extra) or 0


if __name__ == '__main__':
    raise SystemExit(main())
