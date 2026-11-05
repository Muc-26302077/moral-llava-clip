"""Rebuild all six delivery roots from versioned sources without train/test mixing."""
import argparse
import json
from pathlib import Path
import tempfile
import sys

from delivery import ROOT, export, save
sys.path.insert(0, str(ROOT))
from moral_llava import polarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT.parent/'delivery/config.json')
    parser.add_argument('--output', type=Path, required=True, help='New parent directory for the six source_roots')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    raw = ROOT/'data/raw/SMID_moralclip'
    excluded = {r['id'] for r in json.loads((ROOT/'data/excluded_images.json').read_text())['images']}
    reports = {}
    with tempfile.TemporaryDirectory(prefix='smid-delivery-build-') as tmp:
        tmp = Path(tmp)
        polarity.prepare(raw, tmp/'polarity', ROOT/'results/reference/runs/crossmodal-v1/protocol.json',
                         ROOT/'data/excluded_images.json')
        for task in config['tasks']:
            reports[task] = {}
            for split in config['splits']:
                if task == 'polarity-v1':
                    source = tmp/'polarity'/f'{split}.json'
                elif task == 'legacy-three-state-v1':
                    source = tmp/f'legacy-{split}.json'
                    rows = json.loads((ROOT/'data/processed'/f'{split}.json').read_text())
                    save(source, [r for r in rows if r['id'] not in excluded])
                else:
                    raise ValueError(f'Unknown export task: {task}')
                reports[task][split] = export(source, raw, args.output/task/split, config['run_ids'][task])
    save(args.output/'reports/format-validation.json', reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
