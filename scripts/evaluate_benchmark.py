"""Evaluate Qwen3-VL or VLGuard on the same frozen SMID polarity questions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moral_llava import polarity
from moral_llava.common import sha
from scripts.download_models import ASSETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('qwen3-vl-8b','vlguard-posthoc','vlguard-mixed'), required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, help='Pinned LLaVA base, required by VLGuard')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='Smoke only; partial results cannot be scored formally')
    parser.add_argument('--output', type=Path, required=True, help='New raw-answer JSONL')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.limit < 0:
        parser.error('Invalid batch size or limit')
    qwen = args.model == 'qwen3-vl-8b'
    if not qwen and not args.base_model:
        parser.error('VLGuard requires --base-model')
    if args.output.exists():
        raise FileExistsError('Use a new predictions file')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Benchmark generation requires a CUDA GPU')
    source = args.model_dir.expanduser().resolve()
    index = json.loads((source/('model.safetensors.index.json' if qwen else 'pytorch_model.bin.index.json')).read_text())
    shard_hashes = {f:sha(source/f) for f in sorted(set(index['weight_map'].values()))}
    expected = json.loads((ROOT/'models/sources.json').read_text())[args.model]['weights']
    if shard_hashes != expected:
        raise ValueError('Benchmark model weight hashes differ from the pinned public release')
    data = ROOT/'data/polarity-v1/test.json'
    records = polarity.load_records(data, ROOT/'data/raw/SMID_moralclip', require_targets=False)
    if args.limit:
        records = records[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if qwen:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(source,local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        processor.image_processor.size = {'longest_edge':1280*32*32,'shortest_edge':256*32*32}
        model = Qwen3VLForConditionalGeneration.from_pretrained(source,torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa',low_cpu_mem_usage=True,local_files_only=True)
        logits_processors = None
    else:
        from moral_llava.pipeline import resolve_base
        from moral_llava.benchmark_models import load_vlguard
        from transformers import LogitsProcessor, LogitsProcessorList
        base, _ = resolve_base(args.base_model, offline=True)
        model, processor, _ = load_vlguard(args.model, base, source, args.output.with_suffix('.conversion.json'))
        class SuppressAppended(LogitsProcessor):
            def __call__(self, input_ids, scores):
                scores[:,32000:] = -float('inf')
                return scores
        logits_processors = LogitsProcessorList([SuppressAppended()])
    model = model.to('cuda').eval()
    provenance = dict(model=args.model,benchmark_task='ticr',data_sha256=sha(data),precision='bf16',
        decoding='greedy',max_new_tokens=16,batch_size=args.batch_size,smoke_only=bool(args.limit),
        base_model_id=ASSETS[args.model][0],base_revision=ASSETS[args.model][1],
        script_sha256=sha(Path(__file__)),weight_sha256=shard_hashes)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with args.output.open('x',encoding='utf-8') as stream:
        for offset in range(0,len(records),args.batch_size):
            batch = records[offset:offset+args.batch_size]
            if qwen:
                messages = []
                for row in batch:
                    content = []
                    if row['modality']=='image':
                        content.append({'type':'image','image':row['image_path']})
                    question = row['prompt'][6:-11].replace('<image>\n','',1)
                    content.append({'type':'text','text':question})
                    messages.append([{'role':'user','content':content}])
                rendered = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                inputs = processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                    return_dict=True,return_tensors='pt',padding=True)
            else:
                from moral_llava.benchmark_models import inputs_for_vlguard
                system = ('A chat between a curious user and an artificial intelligence assistant. '
                    "The assistant gives helpful, detailed, and polite answers to the user's questions.")
                adapted = [dict(row,prompt=system+' '+row['prompt']) for row in batch]
                rendered = [r['prompt'] for r in adapted]
                inputs = inputs_for_vlguard(processor,adapted)
            inputs = {k:v.to('cuda',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda')
                      for k,v in inputs.items()}
            width = inputs['input_ids'].shape[1]
            if width+16 > (8192 if qwen else 4096):
                raise ValueError('Context limit exceeded; no truncation allowed')
            with torch.inference_mode():
                generated = model.generate(**inputs,do_sample=False,max_new_tokens=16,use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id,logits_processor=logits_processors)
            answers = processor.tokenizer.batch_decode(generated[:,width:],skip_special_tokens=True)
            for row,answer,prompt in zip(batch,answers,rendered):
                result = polarity.format_prediction(row,answer)
                result.update(provenance,rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            stream.flush()
            print(f'{args.model}: {offset+len(batch)}/{len(records)}',flush=True)
    if not args.limit:
        from run_moral_llava import main as entry
        entry(['score','--predictions',str(args.output),'--output',str(args.output.with_suffix('.metrics.json'))])


if __name__ == '__main__':
    main()
