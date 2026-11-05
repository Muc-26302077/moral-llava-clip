"""Measure the existing MoralCLIP-LLaVA training format, including image expansion.

These are current-model statistics, not Qwen3.5-VL receiver acceptance numbers.
No model weights are loaded and no training is performed.
"""
import argparse
from pathlib import Path
import sys

from delivery import ROOT, save, sha
from evaluate_delivery import load_delivery
sys.path.insert(0, str(ROOT))
from moral_llava.binary_runtime import inputs_for_records
from moral_llava.train import build_supervised_sequence, unpad_input_ids


def measure(source_root, processor, *, image_tokens=196, max_length=2048, batch_size=16):
    source_root = Path(source_root)
    rows = load_delivery(source_root)
    records = [{'id': r['id'], 'modality': 'image' if r.get('image') else 'text',
        'image_path': str(source_root/r['image']) if r.get('image') else None,
        'prompt': f'USER: {r["conversations"][0]["value"]} ASSISTANT:',
        'target_json': r['conversations'][-1]['value']} for r in rows]
    lengths, trainable = [], []
    image_id = processor.tokenizer.convert_tokens_to_ids('<image>')
    for start in range(0, len(records), batch_size):
        batch = records[start:start+batch_size]
        processed = inputs_for_records(processor, batch)
        for i, r in enumerate(batch):
            prompt = unpad_input_ids(processed['input_ids'][i].tolist(), processed['attention_mask'][i].tolist())
            answer = processor.tokenizer.encode(' '+r['target_json'], add_special_tokens=False)
            seq = build_supervised_sequence(prompt, answer, eos_token_id=processor.tokenizer.eos_token_id,
                    image_token_id=image_id, expected_image_tokens=image_tokens if r['modality']=='image' else 0,
                    max_length=max_length)
            lengths.append(len(seq['input_ids']))
            trainable.append(sum(x != -100 for x in seq['labels'][1:]))
    ordered = sorted(lengths)
    return {'scope': 'existing-MoralCLIP-LLaVA-training-format', 'receiver_acceptance': 'not_measured',
        'records': len(records), 'counting_policy': 'One dataset pass; no padding; prompt + expanded image + answer + EOS; shifted supervised labels',
        'image_tokens_per_image': image_tokens, 'max_seq_length': max_length,
        'total_tokens': sum(lengths), 'trainable_tokens': sum(trainable),
        'length_distribution': {'min': min(lengths), 'p50': ordered[len(ordered)//2],
            'p95': ordered[min(len(ordered)-1, int(.95*len(ordered)))], 'max': max(lengths)},
        'min_trainable_tokens': min(trainable),
        'records_sha256': sha(source_root/'records.jsonl')}


def main():
    from transformers import AutoProcessor
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--processor', required=True, help='Saved current MoralCLIP-LLaVA processor directory')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.resolve().is_relative_to(args.source_root.resolve()):
        p.error('Statistics must be written outside source_root')
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    processor.tokenizer.padding_side = 'right'
    report = measure(args.source_root, processor)
    report['processor'] = args.processor
    report['tokenizer_sha256'] = sha(Path(args.processor)/'tokenizer.json')
    save(args.output, report)
    print(report)


if __name__ == '__main__':
    main()
