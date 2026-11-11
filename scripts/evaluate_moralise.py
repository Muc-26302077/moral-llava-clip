"""Download/prepare MORALISE, evaluate the baseline, or rescore existing answers."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.moralise import prepare_moralise, score_moralise, read, dump, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','infer','score'))
    parser.add_argument('--base-model', type=Path)
    parser.add_argument('--moralclip-model', type=Path)
    parser.add_argument('--adapter', type=Path, default=ROOT/'models/moralclip-llava-polarity')
    parser.add_argument('--data', type=Path, default=ROOT/'data/moralise-native.json')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id='Ze1025/MORALISE',repo_type='dataset',
            revision='80ed627e8f12c2e8ab8839f5639d56d4625b4127',local_dir=str(ROOT/'data/moralise'),
            allow_patterns=['M1/**','M1-Right/**','M1-all/**','M2/**','M2-Right/**','M2-all/**','README.md'])
        prepare_moralise()
        return
    if not args.output:
        parser.error('--output is required')
    if args.output.exists():
        raise FileExistsError('Use a new output file')
    rows = read(args.data)
    if args.command == 'score':
        if not args.predictions:
            parser.error('score requires --predictions')
        preds = [json.loads(s) for s in args.predictions.read_text().splitlines() if s.strip()]
        if any(p.get('smoke_only') for p in preds):
            raise ValueError('Cannot formally score smoke predictions')
        result = score_moralise(rows,preds)
        dump(args.output,result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    if not args.base_model or not args.moralclip_model:
        parser.error('infer requires --base-model and --moralclip-model')
    if args.batch_size <= 0 or args.limit < 0:
        parser.error('Invalid batch size or limit')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('MORALISE generation requires a CUDA GPU')
    from transformers import AutoProcessor,LlavaForConditionalGeneration
    from peft import PeftModel
    from moral_llava.pipeline import resolve_base
    from moral_llava.moralclip_integration import install_vision_tower,validate_adapter_metadata
    from moral_llava.train import configure_processor
    from moral_llava.polarity_runtime import inputs_for_records
    base,_ = resolve_base(args.base_model,offline=True)
    processor = AutoProcessor.from_pretrained(args.adapter,local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(base,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',low_cpu_mem_usage=True,local_files_only=True)
    integration = install_vision_tower(model,processor,args.moralclip_model,local_files_only=True)
    validate_adapter_metadata(read(args.adapter/'training_metadata.json'),integration)
    configure_processor(processor,model.config,padding_side='left')
    model = PeftModel.from_pretrained(model,str(args.adapter),is_trainable=False).to('cuda').eval()
    if args.limit:
        rows=rows[:args.limit]
    provenance = {'data_sha256':sha(args.data),'adapter_sha256':sha(args.adapter/'adapter_model.safetensors'),
        'benchmark_task':'moralise','precision':'bf16','decoding':'greedy','max_new_tokens':64,
        'batch_size':args.batch_size,'script_sha256':sha(Path(__file__)),'smoke_only':bool(args.limit)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with args.output.open('x',encoding='utf-8') as stream:
        for start in range(0,len(rows),args.batch_size):
            batch=rows[start:start+args.batch_size]
            adapted=[dict(row,modality='image',prompt='USER: <image>\n'+row['question']+' ASSISTANT:') for row in batch]
            inputs=inputs_for_records(processor,adapted)
            inputs={k:v.to('cuda',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k,v in inputs.items()}
            width=inputs['input_ids'].shape[1]
            if width+64 > 4096:
                raise ValueError('Context limit exceeded; no truncation allowed')
            with torch.inference_mode():
                generated=model.generate(**inputs,do_sample=False,max_new_tokens=64,use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id)
            answers=processor.tokenizer.batch_decode(generated[:,width:],skip_special_tokens=True)
            for row,answer in zip(batch,answers):
                stream.write(json.dumps(dict(id=row['id'],raw_prediction=answer,**provenance),ensure_ascii=False)+'\n')
            stream.flush()
            print(f'{start+len(batch)}/{len(rows)}',flush=True)


if __name__ == '__main__':
    main()
