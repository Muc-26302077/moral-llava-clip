"""Acquire pinned public model files; also verify an existing local directory."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = {
    'moralclip': ('anaaa2/moralclip-base', '1e337922befebde3f0cadb5749f37ae9521a0751', ['*.json','*.txt','*.safetensors']),
    'llava': ('llava-hf/llava-1.5-7b-hf', 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9', ['*.json','*.model','*.safetensors']),
    'qwen3-vl-8b': ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', ['*.json','*.safetensors','*.txt']),
    'vlguard-posthoc': ('ys-zong/llava-v1.5-7b-Posthoc', '3529d19cb2e919358b317ac308dba37effcc8b5b', ['config.json','generation_config.json','pytorch_model*','tokenizer*','special_tokens_map.json']),
    'vlguard-mixed': ('ys-zong/llava-v1.5-7b-Mixed', '757221ed23a5bec4bdaa885a967f37ee9bd5cd21', ['config.json','generation_config.json','pytorch_model*','tokenizer*','special_tokens_map.json']),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=ASSETS)
    parser.add_argument('--output', type=Path, help='Download directory; default models/base/<model>')
    parser.add_argument('--verify-local', type=Path, help='Verify local files without downloading')
    args = parser.parse_args()
    repo, revision, patterns = ASSETS[args.model]
    if args.verify_local:
        path = args.verify_local.expanduser().resolve()
    else:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=patterns,
                    local_dir=str(args.output or ROOT/'models/base'/args.model))).resolve()
    if args.model == 'llava':
        from moral_llava.pipeline import resolve_base
        path, checks = resolve_base(path, offline=True)
    elif args.model == 'moralclip':
        from moral_llava.moralclip_integration import sha, WEIGHT_SHA256
        if sha(path/'model.safetensors') != WEIGHT_SHA256:
            raise ValueError('MoralCLIP SHA-256 mismatch')
        checks = {'weight_sha256': WEIGHT_SHA256}
    else:
        index = path/('model.safetensors.index.json' if args.model=='qwen3-vl-8b' else 'pytorch_model.bin.index.json')
        shards = set(json.loads(index.read_text())['weight_map'].values())
        from moral_llava.common import sha
        checks = {name: sha(path/name) for name in sorted(shards)}
        if any((path/name).stat().st_size < 1_000_000 for name in shards):
            raise ValueError('Missing model shards or unresolved Git LFS files')
        expected = json.loads((ROOT/'models/sources.json').read_text())[args.model]['weights']
        if checks != expected:
            raise ValueError('Weight SHA-256 differs from the pinned public model')
    print(json.dumps({'model':repo,'revision':revision,'path':str(path),'checks':checks}, indent=2))


if __name__ == '__main__':
    main()
