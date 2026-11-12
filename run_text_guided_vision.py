"""GPU training and genuine generative TICR evaluation of a guided encoder.

Commands: preflight (CPU-safe), smoke (small train/validation only), experiment
(full validation selection, then one final test), report (existing results).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
BASE_REVISION = 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
def baseline_sha(args):
    if not hasattr(args, '_baseline_sha256'):
        args._baseline_sha256 = sha(args.baseline_adapter/'adapter_model.safetensors')
    return args._baseline_sha256


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def ensure_data(args):
    from moral_llava import polarity
    if not args.data.is_dir():
        import contextlib
        import io
        from run_moral_llava import main as prepare
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(['prepare', '--output', str(args.data), '--image-root', str(args.image_root)])
    audit = polarity.read(args.data / 'audit.json')
    raw = {s: polarity.read(args.data / f'{s}.json') for s in ('train', 'validation', 'test')}
    expected = {'train': 9806, 'validation': 580, 'test': 576}
    for split, rows in raw.items():
        if len(rows) != expected[split] or sha(args.data / f'{split}.json') != audit['split_record_sha256'][split]:
            raise ValueError('Frozen dataset count/hash mismatch: ' + split)
    scenes = {s: {r['scene_id'] for r in rows} for s, rows in raw.items()}
    if any(scenes[a] & scenes[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Cross-split scene leakage')
    normalized = {s: polarity.load_records(args.data / f'{s}.json', args.image_root,
                                          require_targets=(s != 'test')) for s in raw}
    return raw, normalized, audit


def balanced_subset(rows, per_class, *, paired=False):
    from moral_llava import polarity
    rng = random.Random(42)
    chosen = set()
    for task in polarity.TASKS:
        for positive in (False, True):
            pool = [r for r in rows if r['task']==task and r['modality']=='image' and
                    polarity.parse_answer(r['target_json'], task) is positive]
            rng.shuffle(pool)
            if len(pool) < per_class:
                raise ValueError('Insufficient sample support')
            chosen.update((r['scene_id'],task) for r in pool[:per_class])
    return [r for r in rows if (r['scene_id'],r['task']) in chosen and (paired or r['modality']=='image')]


def preflight(args):
    import torch
    from moral_llava.moralclip_integration import WEIGHT_SHA256
    if not args.base_model.is_dir():
        raise ValueError('Use the pinned LLaVA snapshot directory: ' + BASE_REVISION)
    index = json.loads((args.base_model / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    for shard in shards:
        path = args.base_model / shard
        if not path.is_file() or path.stat().st_size < 1000000:
            raise ValueError('Missing base model shard / Git LFS pointer: ' + str(path))
    baseline_weight=args.baseline_adapter/'adapter_model.safetensors'
    if baseline_weight.stat().st_size<1000000:
        raise ValueError('Baseline weights are a Git LFS pointer; run git lfs pull before GPU training')
    if args.expected_baseline_sha256 and sha(baseline_weight) != args.expected_baseline_sha256:
        raise ValueError('Baseline adapter differs from the explicitly required SHA-256')
    if sha(args.moralclip_model / 'model.safetensors') != WEIGHT_SHA256:
        raise ValueError('Original MoralCLIP weight mismatch')
    raw, normalized, audit = ensure_data(args)
    report = {'cuda_available': torch.cuda.is_available(), 'torch':torch.__version__,
              'base_revision':BASE_REVISION, 'baseline_adapter_sha256':baseline_sha(args),
              'base_snapshot':str(args.base_model.resolve()),
              'question_counts':{s:len(r) for s,r in raw.items()},
              'train_image_questions':sum(r['modality']=='image' for r in normalized['train']),
              'conditioning':'human question only; no paired caption or answer',
              'training':'GPU generation loss backpropagates into new visual encoder residual only',
              'cpu_small_sample':'separate real-image linear-probe diagnostic, not TICR',
              'gpu_training_executed':False}
    for name in ('memory.max','cpu.max'):
        p=Path('/sys/fs/cgroup')/name
        if p.exists(): report[name]=p.read_text().strip()
    if torch.cuda.is_available():
        report['gpu_name']=torch.cuda.get_device_name(0)
        report['gpu_memory_gib']=torch.cuda.get_device_properties(0).total_memory / 2**30
        report['bf16_supported']=torch.cuda.is_bf16_supported()
    return report, raw, normalized, audit


def load_runtime(args, *, guidance_installer=None):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel
    from moral_llava.train import configure_processor
    from moral_llava.moralclip_integration import install_vision_tower, validate_adapter_metadata
    from moral_llava.text_guided_vision import GuidanceConfig, install_guidance
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. GPU smoke/training did not start; run preflight on CPU.')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported; choose --precision fp32 explicitly')
    from moral_llava.pipeline import resolve_base
    resolve_base(args.base_model, offline=True)  # verify all pinned shard hashes
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    metadata=json.loads((args.baseline_adapter/'training_metadata.json').read_text())
    processor=AutoProcessor.from_pretrained(args.baseline_adapter,local_files_only=True)
    model=LlavaForConditionalGeneration.from_pretrained(args.base_model, local_files_only=True,
          torch_dtype=dtype, attn_implementation='sdpa', low_cpu_mem_usage=True, device_map={'':'cuda'})
    integration=install_vision_tower(model,processor,args.moralclip_model,local_files_only=True)
    validate_adapter_metadata(metadata,integration)
    model=PeftModel.from_pretrained(model,args.baseline_adapter,is_trainable=False)
    model.eval()
    if guidance_installer is None:
        cfg=GuidanceConfig(vision_dim=model.config.vision_config.hidden_size,
                           text_dim=model.config.text_config.hidden_size,rank=args.rank)
        block,index=install_guidance(model,cfg)
    else:
        block,index=guidance_installer(model,args.rank)
    configure_processor(processor,model.config,padding_side='right')
    # Non-reentrant checkpointing retains gradients through the frozen decoder.
    model.get_base_model().language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    return model,processor,block,index,integration,dtype


def move_batch(batch, dtype):
    return {k:v.to(device='cuda',dtype=dtype if v.is_floating_point() else v.dtype) for k,v in batch.items()}


def predictions(args, model, processor, block, dtype, records, *, enabled, guidance_hash, split, smoke):
    import torch
    from moral_llava import polarity
    from moral_llava.polarity_runtime import inputs_for_records
    from moral_llava.text_guided_vision import METHOD, question_conditions
    from moral_llava.moralclip_integration import METHOD as BASE_METHOD, WEIGHT_SHA256
    from moral_llava.train import unpad_input_ids
    processor.tokenizer.padding_side='left'
    model.eval()
    output=[]
    for offset in range(0,len(records),args.eval_batch_size):
        current=records[offset:offset+args.eval_batch_size]
        image_rows=[r for r in current if r['modality']=='image']
        conditions=question_conditions(model,processor.tokenizer,image_rows) if image_rows and enabled else None
        inputs=inputs_for_records(processor,current)
        for i,row in enumerate(current):
            ids=unpad_input_ids(inputs['input_ids'][i].tolist(),inputs['attention_mask'][i].tolist())
            expected=196 if row['modality']=='image' else 0
            if ids.count(model.config.image_token_index)!=expected or len(ids)+16>args.max_length:
                raise ValueError('Expanded image tokens/context budget invalid: '+row['id'])
        inputs=move_batch(inputs,dtype)
        width=inputs['input_ids'].shape[1]
        with torch.inference_mode(),block.conditioned(conditions,enabled=enabled and bool(image_rows)):
            generated=model.generate(**inputs,max_new_tokens=16,do_sample=False,use_cache=True,
                       pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
        raw=processor.batch_decode(generated[:,width:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        for row,text in zip(current,raw):
            p=polarity.format_prediction(row,text)
            p.update(data_sha256=sha(args.data/f'{split}.json'),smoke_only=smoke,decoding='free',
                     base_model_id='llava-hf/llava-1.5-7b-hf',base_revision=BASE_REVISION,
                     adapter_sha256=baseline_sha(args),precision=str(dtype),max_new_tokens=16,
                     vision_weight_sha256=WEIGHT_SHA256,integration_method=METHOD if enabled else BASE_METHOD,
                     guidance_sha256=guidance_hash if enabled else None,
                     condition_source='human_question_frozen_input_embeddings' if enabled else None,
                     guidance_enabled=enabled)
            output.append(p)
        print(f'{split} generated {min(offset+len(current),len(records))}/{len(records)} guided={enabled}',flush=True)
    return output


def write_predictions(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def metrics_for(raw, predictions, audit=None, *, diagnostic=False):
    from moral_llava import polarity
    if diagnostic:
        # The scorer refuses smoke records by design. Internal diagnostics are
        # scored on copies and explicitly marked non-formal, never published as test.
        predictions=[dict(p,smoke_only=False) for p in predictions]
    result=polarity.score(raw,predictions,audit)
    result['formal_test_result']=bool(not diagnostic and audit and result['complete_grouped_benchmark'])
    return result


def selection_key(metrics, baseline):
    m,b=metrics['overall']['primary_macro'],baseline['overall']['primary_macro']
    valid=sum(t['image']['invalid_outputs'] for t in metrics['overall']['per_task'].values())
    baseline_invalid=sum(t['image']['invalid_outputs'] for t in baseline['overall']['per_task'].values())
    if m['image_accuracy']+1e-12<b['image_accuracy'] or valid>baseline_invalid:
        return None
    return m['joint_accuracy'],m['image_balanced_accuracy'],m['ticr']


def experiment(args):
    import torch
    from moral_llava import polarity
    from moral_llava.polarity_runtime import PolaritySFTCollator
    from moral_llava.text_guided_vision import (
        METHOD, question_conditions, save_guidance, load_guidance, verify_original_vision,
    )
    # Fail before creating an experiment directory or trying to load 7B on CPU.
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU; no GPU experiment was started.')
    if args.output is None:
        raise ValueError('--output is required')
    env,raw,rows,audit=preflight(args)
    args.output.mkdir(parents=True,exist_ok=False)
    state={'status':'starting','command':args.command,'started_at':now(),'pid':os.getpid()}
    save(args.output/'status.json',state)
    started=time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model,processor,block,index,integration,dtype=load_runtime(args)
        smoke=args.command=='smoke'
        training=balanced_subset(rows['train'],2) if smoke else [r for r in rows['train'] if r['modality']=='image']
        validation=balanced_subset(rows['validation'],1,paired=True) if smoke else rows['validation']
        selected_ids={r['id'] for r in validation}
        val_raw=[r for r in raw['validation'] if r['id'] in selected_ids]
        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('.adapter.' not in n for n in trainable):
            raise ValueError('Only new visual adapter may train')
        code=[Path(__file__).resolve(),ROOT/'moral_llava/text_guided_vision.py',ROOT/'moral_llava/polarity.py',
              ROOT/'moral_llava/polarity_runtime.py',ROOT/'moral_llava/train.py']
        code += [ROOT/'moral_llava/guided_report.py',ROOT/'moral_llava/moralclip_integration.py',
                 ROOT/'moral_llava/common.py',ROOT/'moral_llava/schema.py',ROOT/'moral_llava/pipeline.py']
        config={'method':METHOD,'guidance_config':asdict(block.adapter.config),
                'wrapped_block_index':index,'environment':env,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'training_examples':len(training),'validation_questions':len(validation),'trainable_names':trainable,
                'trainable_parameters':sum(p.numel() for p in block.adapter.parameters()),
                'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code},
                'data_sha256':{s:sha(args.data/f'{s}.json') for s in raw},
                'test_policy':'no test predictions until validation selection is locked',
                'selection':'image accuracy and valid-rate guards, then joint accuracy / image BA / TICR; earlier ties'}
        save(args.output/'frozen_config.json',config)
        state.update(status='running',phase='baseline_validation');save(args.output/'status.json',state)
        baseline=predictions(args,model,processor,block,dtype,validation,enabled=False,guidance_hash=None,split='validation',smoke=smoke)
        write_predictions(args.output/'baseline-validation.jsonl',baseline)
        baseline_metrics=metrics_for(val_raw,baseline,diagnostic=smoke)
        save(args.output/'baseline-validation-metrics.json',baseline_metrics)
        # Exact zero-init equivalence under the actual 7B decoder, all tasks.
        zero=predictions(args,model,processor,block,dtype,validation[:20],enabled=True,guidance_hash='zero-init',split='validation',smoke=True)
        if [p['raw_prediction'] for p in zero]!=[p['raw_prediction'] for p in baseline[:20]]:
            raise ValueError('Zero-init adapter changed baseline generation')
        initial=save_guidance(args.output/'epoch-0',block,{'epoch':0,'baseline_adapter_sha256':baseline_sha(args)})
        best_path=args.output/'epoch-0'
        best_key=selection_key(baseline_metrics,baseline_metrics)
        optimizer=torch.optim.AdamW(block.adapter.parameters(),lr=args.learning_rate,weight_decay=0)
        collator=PolaritySFTCollator(processor,image_token_id=model.config.image_token_index,
                    expected_image_tokens=model.config.image_seq_length,max_length=args.max_length)
        losses=[]
        max_epochs=1 if smoke else args.epochs
        step=0
        training_started=time.monotonic()
        for epoch in range(1,max_epochs+1):
            shuffled=list(training)
            random.Random(args.seed+epoch).shuffle(shuffled)
            if smoke: shuffled=shuffled[:args.smoke_steps]
            optimizer.zero_grad()
            model.eval()  # frozen LoRA dropout must also remain disabled
            # Activate LlamaModel's checkpoint branch without enabling frozen
            # LoRA/attention dropout in its child modules.
            model.get_base_model().language_model.model.training=True
            block.adapter.train()
            processor.tokenizer.padding_side='right'
            batches=[shuffled[i:i+args.train_batch_size] for i in range(0,len(shuffled),args.train_batch_size)]
            for batch_index,records in enumerate(batches):
                position=batch_index*args.train_batch_size
                condition=question_conditions(model,processor.tokenizer,records)
                inputs=move_batch(collator(records),dtype)
                with block.conditioned(condition):
                    output=model(**inputs,use_cache=False)
                    penalty=block.last_penalty
                    # Match per-example supervision when answers differ in
                    # token length, including in a partial final batch.
                    shifted_logits=output.logits[:,:-1].float()
                    shifted_labels=inputs['labels'][:,1:]
                    token_losses=torch.nn.functional.cross_entropy(
                        shifted_logits.reshape(-1,shifted_logits.shape[-1]),
                        shifted_labels.reshape(-1),ignore_index=-100,reduction='none').view_as(shifted_labels)
                    generation_loss=(token_losses.sum(1)/(shifted_labels!=-100).sum(1)).mean()
                    loss=generation_loss+args.preservation_weight*penalty
                    if not torch.isfinite(loss):raise ValueError('Non-finite generation loss')
                    group_start=(batch_index//args.gradient_accumulation)*args.gradient_accumulation
                    group_examples=sum(len(batch) for batch in batches[group_start:group_start+args.gradient_accumulation])
                    (loss*len(records)/group_examples).backward()
                if (batch_index+1)%args.gradient_accumulation==0 or position+len(records)==len(shuffled):
                    torch.nn.utils.clip_grad_norm_(block.adapter.parameters(),1.0)
                    if not any(p.grad is not None and p.grad.abs().sum()>0 for p in block.adapter.parameters()):
                        raise ValueError('No gradient reached the visual encoder adapter')
                    optimizer.step();optimizer.zero_grad();step+=1
                losses.append({'epoch':epoch,'example':position+len(records),'batch_size':len(records),'optimizer_step':step,
                               'generation_loss':float(generation_loss.detach()),'preservation':float(penalty.detach()),
                               'elapsed_training_seconds':time.monotonic()-training_started})
                if batch_index%10==0:
                    state.update(phase='training',epoch=epoch,example=position+len(records),total=len(shuffled),optimizer_step=step)
                    save(args.output/'status.json',state);save(args.output/'training-loss.json',losses)
                    print(json.dumps(losses[-1]),flush=True)
            checkpoint=args.output/f'epoch-{epoch}'
            saved=save_guidance(checkpoint,block,{'epoch':epoch,'baseline_adapter_sha256':baseline_sha(args)})
            outputs=predictions(args,model,processor,block,dtype,validation,enabled=True,
                         guidance_hash=saved['weights_sha256'],split='validation',smoke=smoke)
            write_predictions(args.output/f'epoch-{epoch}-validation.jsonl',outputs)
            metrics=metrics_for(val_raw,outputs,diagnostic=smoke)
            save(args.output/f'epoch-{epoch}-validation-metrics.json',metrics)
            candidate=selection_key(metrics,baseline_metrics)
            if candidate is not None and candidate>best_key:
                best_path,best_key=checkpoint,candidate
        save(args.output/'training-loss.json',losses)
        metadata=load_guidance(best_path,block)
        save(args.output/'selection.json',{'checkpoint':str(best_path.resolve()),'epoch':metadata['metadata']['epoch'],
                    'guidance_sha256':metadata['weights_sha256'],'validation_key':best_key,'locked_at':now()})
        # Verify serialization using actual generation after restoring selected adapter.
        reload_predictions=predictions(args,model,processor,block,dtype,validation[:20],enabled=True,
                          guidance_hash=metadata['weights_sha256'],split='validation',smoke=True)
        expected=baseline if best_path.name=='epoch-0' else [json.loads(line) for line in
                   (args.output/f'{best_path.name}-validation.jsonl').read_text().splitlines()]
        if [p['raw_prediction'] for p in reload_predictions]!=[p['raw_prediction'] for p in expected[:20]]:
            raise ValueError('Selected checkpoint reload changed actual generation')
        verified=verify_original_vision(model,integration)
        if not smoke:
            state.update(phase='final_test');save(args.output/'status.json',state)
            # Baseline and new model use identical precision/input/decoding.
            for name,enabled in [('baseline',False),('guided',True)]:
                outputs=predictions(args,model,processor,block,dtype,rows['test'],enabled=enabled,
                       guidance_hash=metadata['weights_sha256'] if enabled else None,split='test',smoke=False)
                write_predictions(args.output/f'{name}-test.jsonl',outputs)
                metrics=metrics_for(raw['test'],outputs,audit)
                if not metrics['formal_test_result']:raise ValueError('Incomplete formal test')
                save(args.output/f'{name}-test-metrics.json',metrics)
            # A structural ablation uses the same trained weights with the
            # residual disabled. Text branch is unchanged by construction.
            from moral_llava.guided_report import build_report
            build_report(args.output,raw['test'])
        for name,checksum in config['source_sha256'].items():
            if sha(ROOT/name)!=checksum:raise ValueError('Source changed during experiment: '+name)
        state.update(status='complete',phase='smoke_complete' if smoke else 'formal_complete',
                     finished_at=now(),seconds=time.monotonic()-started,
                     frozen_visual_tensors_verified=verified,
                     cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                     formal_ticr_available=not smoke,optimizer_step=step,example=len(shuffled))
        save(args.output/'status.json',state)
        print(json.dumps(state,ensure_ascii=False,indent=2))
    except BaseException as error:
        state.update(status='failed',error=str(error),finished_at=now());save(args.output/'status.json',state)
        raise


def infer_saved(args):
    """Evaluate a saved trained or selected module with its paired baseline."""
    if not args.output or not args.guidance:
        raise ValueError('infer requires --output and --guidance')
    preflight_report, raw, rows, audit = preflight(args)
    args.output.mkdir(parents=True, exist_ok=False)
    args.rank = json.loads((args.guidance/'guidance.json').read_text())['config']['rank']
    model, processor, block, _, integration, dtype = load_runtime(args)
    from moral_llava.text_guided_vision import load_guidance, verify_original_vision
    metadata = load_guidance(args.guidance, block)
    if metadata['metadata']['baseline_adapter_sha256'] != baseline_sha(args):
        raise ValueError('Guidance was trained against a different baseline adapter')
    all_predictions = {}
    for name, enabled in (('baseline',False),('guided',True)):
        outputs = predictions(args,model,processor,block,dtype,rows['test'],enabled=enabled,
            guidance_hash=metadata['weights_sha256'] if enabled else None,split='test',smoke=False)
        all_predictions[name] = {p['id']:p for p in outputs}
        write_predictions(args.output/f'{name}-test.jsonl',outputs)
        save(args.output/f'{name}-metrics.json',metrics_for(raw['test'],outputs,audit))
    if any(p['raw_prediction'] != all_predictions['baseline'][key]['raw_prediction']
           for key,p in all_predictions['guided'].items() if p['modality']=='text'):
        raise ValueError('Frozen text branch changed')
    save(args.output/'verification.json',{'baseline_sha256':baseline_sha(args),
        'guidance_sha256':metadata['weights_sha256'],
        'frozen_visual_tensors_verified':verify_original_vision(model,integration)})
    print('Results:',args.output)


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','smoke','experiment','infer','report'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--base-model',type=Path,default=Path(os.environ.get('BASE_SNAPSHOT',
        str(ROOT/'models/base/llava'))))
    parser.add_argument('--baseline-adapter',type=Path,default=ROOT/'models/moralclip-llava-polarity')
    parser.add_argument('--moralclip-model',type=Path,default=Path(os.environ.get('MORALCLIP_MODEL',str(ROOT/'models/base/moralclip'))))
    parser.add_argument('--expected-baseline-sha256', help='Optional pinned adapter hash; omit for a newly trained baseline')
    parser.add_argument('--guidance',type=Path,help='Saved guidance directory for infer')
    parser.add_argument('--data',type=Path,default=ROOT/'data/polarity-v1')
    parser.add_argument('--image-root',type=Path,default=ROOT/'data/raw/SMID_moralclip')
    parser.add_argument('--precision',choices=('bf16','fp32'),default='bf16')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--rank',type=int,default=32)
    parser.add_argument('--learning-rate',type=float,default=5e-4)
    parser.add_argument('--preservation-weight',type=float,default=.05)
    parser.add_argument('--gradient-accumulation',type=int,default=1)
    parser.add_argument('--train-batch-size',type=int,default=8)
    parser.add_argument('--eval-batch-size',type=int,default=1)
    parser.add_argument('--max-length',type=int,default=2048)
    parser.add_argument('--smoke-steps',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    return parser


def main(argv=None):
    parser=make_parser();args=parser.parse_args(argv)
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
    if min(args.epochs,args.rank,args.gradient_accumulation,args.train_batch_size,args.eval_batch_size,args.max_length,args.smoke_steps)<=0:
        parser.error('Counts and dimensions must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or not math.isfinite(args.preservation_weight) or args.preservation_weight<0:
        parser.error('Invalid learning rate or preservation weight')
    if args.command=='preflight':
        result,*_=preflight(args)
        if args.output:save(args.output/'preflight.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command=='infer':
        infer_saved(args)
    elif args.command=='report':
        if not args.output:parser.error('--output is required')
        from moral_llava.guided_report import build_report
        data=json.loads((args.data/'test.json').read_text())
        build_report(args.output,data)
    else:
        experiment(args)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,RuntimeError,OSError) as error:
        print(f'Guided encoder: {error}',file=sys.stderr)
        raise SystemExit(2)
