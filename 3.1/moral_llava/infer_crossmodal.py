"""Evaluate frozen base/old LoRA on fixed image/text pairs. Never trains."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

from .crossmodal import (ROOT, REVISION, AnswerTrie, read_json, score, sha256,
                         validate_protocol, write_new_json)
from .infer import format_prediction

def now():
    return datetime.now(timezone.utc).isoformat()


def update_status(path, record):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def run(args):
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from .train import configure_processor

    protocol = validate_protocol(read_json(args.protocol))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. No model was loaded; mount a GPU or explicitly select --device cpu.')
    if args.device == 'cuda' and args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA device does not support BF16')
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    if not (args.adapter/'adapter_model.safetensors').is_file():
        raise FileNotFoundError('Missing old LoRA adapter')
    metadata = read_json(args.adapter/'training_metadata.json')
    if metadata['resolved_base_commit'] != REVISION:
        raise ValueError('Adapter revision mismatch')
    # A custom local folder must explicitly retain the same recorded revision.
    if args.base_model.name != REVISION:
        raise ValueError('Use the revision-named local snapshot to prevent base-model ambiguity')
    for scene in protocol['scenes']:
        if sha256(scene['image']) != scene['image_sha256']:
            raise ValueError(f"Image changed: {scene['id']}")
        if hashlib.sha256(scene['caption'].encode()).hexdigest() != scene['caption_sha256']:
            raise ValueError('Caption changed')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'loading', 'started_at': now(), 'protocol_sha256': sha256(args.protocol),
        'torch': torch.__version__, 'python': platform.python_version(), 'device': args.device,
        'precision': args.precision, 'base_revision': REVISION, 'training': False, 'output_repair': False,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_sha256': {p.name: sha256(p) for p in [Path(__file__), ROOT/'moral_llava/crossmodal.py',
                                                                   ROOT/'moral_llava/schema.py']},
        'adapter_sha256': sha256(args.adapter/'adapter_model.safetensors')}
    status_path = args.output/'status.json'
    update_status(status_path, record)
    try:
        import transformers, peft
        record.update({'transformers': transformers.__version__, 'peft': peft.__version__})
        started = time.monotonic()
        processor = AutoProcessor.from_pretrained(str(args.base_model), local_files_only=True)
        model = LlavaForConditionalGeneration.from_pretrained(str(args.base_model), torch_dtype=dtype,
            attn_implementation='sdpa', local_files_only=True, low_cpu_mem_usage=True).to(args.device).eval()
        expected_tokens = configure_processor(processor, model.config, padding_side='left')
        record['model_load_seconds'] = time.monotonic()-started
        record['status'] = 'running'
        update_status(status_path, record)
        scenes = protocol['scenes'][:args.limit] if args.limit else protocol['scenes']
        decoding_modes = ('free', 'constrained') if args.decoding == 'both' else (args.decoding,)
        models = ('base', 'lora') if args.models == 'both' else (args.models,)
        all_predictions = {}
        for model_name in models:
            if model_name == 'lora':
                model = peft.PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False).eval()
            for decoding in decoding_modes:
                trie = AnswerTrie(processor.tokenizer) if decoding == 'constrained' else None
                if trie and trie.max_length > args.max_new_tokens:
                    raise ValueError(f'Constrained JSON needs {trie.max_length} tokens; increase max-new-tokens')
                predictions = []
                with (args.output/f'{model_name}_{decoding}.jsonl').open('x', encoding='utf-8') as handle:
                    for modality in ('image', 'text'):
                        for offset in range(0, len(scenes), args.batch_size):
                            batch = scenes[offset:offset+args.batch_size]
                            prompts = [protocol['image_prompt'] if modality == 'image' else
                                       protocol['text_prompt_template'].replace('{caption}', s['caption']) for s in batch]
                            if modality == 'image':
                                images = []
                                for s in batch:
                                    with Image.open(s['image']) as im:
                                        images.append(ImageOps.exif_transpose(im).convert('RGB'))
                                inputs = processor(text=prompts, images=images, padding=True,
                                                   truncation=False, return_tensors='pt')
                            else:
                                inputs = processor.tokenizer(prompts, padding=True, truncation=False, return_tensors='pt')
                            counts = (inputs['input_ids'] == model.config.image_token_index).sum(-1).tolist()
                            if any(n != (expected_tokens if modality == 'image' else 0) for n in counts):
                                raise ValueError('Wrong image-token count')
                            width = inputs['input_ids'].shape[1]
                            if width + args.max_new_tokens > 2048:
                                raise ValueError('Context overflow; no input was truncated')
                            inputs = {k: v.to(device=args.device, dtype=dtype) if v.is_floating_point()
                                      else v.to(args.device) for k, v in inputs.items()}
                            extra = {}
                            if trie:
                                extra['prefix_allowed_tokens_fn'] = lambda batch_id, ids: trie.allowed(ids[width:].tolist())
                            batch_start = time.monotonic()
                            with torch.inference_mode():
                                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, num_beams=1, use_cache=True,
                                    pad_token_id=processor.tokenizer.pad_token_id,
                                    eos_token_id=processor.tokenizer.eos_token_id, **extra)
                            for s, prompt, tokens in zip(batch, prompts, generated[:, width:]):
                                token_ids, eos = tokens.tolist(), processor.tokenizer.eos_token_id
                                ended = eos in token_ids
                                count = token_ids.index(eos)+1 if ended else len(token_ids)
                                raw = processor.tokenizer.decode(tokens, skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False)
                                row = format_prediction(s['id'], raw)
                                row.update({'model': model_name, 'modality': modality, 'decoding': decoding,
                                    'group': s['group'], 'protocol_sha256': record['protocol_sha256'],
                                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                                    'generated_tokens_including_eos': count, 'ended_with_eos': ended,
                                    'caption_index': s['caption_index'] if modality == 'text' else None})
                                if trie and row['parse_error']:
                                    raise RuntimeError('Constrained generation produced an invalid answer')
                                predictions.append(row)
                                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                            handle.flush()
                            print(f'{model_name}/{decoding}/{modality} '
                                  f'{min(offset+len(batch), len(scenes))}/{len(scenes)} '
                                  f'batch_seconds={time.monotonic()-batch_start:.2f}', flush=True)
                all_predictions[model_name, decoding] = predictions
        if not args.limit and args.models == 'both':
            for decoding in decoding_modes:
                report = score(protocol, {n: all_predictions[n, decoding] for n in ('base', 'lora')})
                report['run_status'] = 'complete'
                write_new_json(args.output/f'metrics_{decoding}.json', report)
        record.update({'status': 'smoke_complete' if args.limit else 'complete', 'finished_at': now(),
                       'benchmark_complete': not args.limit and args.models == 'both',
                       'total_seconds': time.monotonic()-started})
        update_status(status_path, record)
    except Exception as error:
        record.update({'status': 'failed', 'error': str(error), 'finished_at': now()})
        update_status(status_path, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'results/reference/runs/crossmodal-v1/protocol.json')
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, default=ROOT/'models/moralllava')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--decoding', choices=('free', 'constrained', 'both'), default='both')
    parser.add_argument('--models', choices=('base', 'lora', 'both'), default='both')
    parser.add_argument('--limit', type=int, default=0, help='Smoke only: no benchmark metrics are written')
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.cpu_threads) < 1 or args.limit < 0:
        parser.error('Invalid batch size, token budget, CPU threads or limit')
    run(args)


if __name__ == '__main__':
    main()
