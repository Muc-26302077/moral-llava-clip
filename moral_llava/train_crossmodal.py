"""Continue old LoRA with paired image/text SFT and label-state JS alignment.

Reads only train/validation manifests. Test evaluation is a separate command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import re
import time

from .crossmodal import ROOT, REVISION, AnswerTrie, image_prompt, text_prompt, read_json, sha256
from .schema import FOUNDATION_KEYS, STATES, labels_to_target, parse_prediction
from .train import configure_processor, pad_supervised_batch, unpad_input_ids

BASE = Path('/root/autodl-tmp/cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots')/REVISION


def dump(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def load_splits(root):
    original = read_json(root/'smid_manifest.json')
    splits = {'train': [], 'validation': []}
    ids, hashes = set(), {}
    for r in original:
        if r['split'] not in splits:
            raise ValueError('Training manifest unexpectedly contains another split')
        if r['id'] in ids:
            raise ValueError('Duplicate source ID')
        ids.add(r['id'])
        image = (root/r['image_path']).resolve()
        if not image.is_relative_to(root.resolve()) or not image.is_file():
            raise ValueError('Invalid image path')
        h = sha256(image)
        if h in hashes and hashes[h] != r['split']:
            raise ValueError('Exact image bytes cross train/validation')
        hashes[h] = r['split']
        target = labels_to_target(r['moral_labels'])
        if not r['captions'] or any(not isinstance(c,str) or not c.strip() or '<image>' in c for c in r['captions']):
            raise ValueError('Invalid captions')
        splits[r['split']].append({'id':r['id'], 'image':str(image), 'captions':r['captions'], 'target':target})
    for split, rows in splits.items():
        rows.sort(key=lambda r:r['id'])
        processed = read_json(ROOT/f'data/processed/{split}.json')
        references = {r['id']:parse_prediction(r['conversations'][1]['value']) for r in processed}
        if {r['id'] for r in rows} != set(references) or any(r['target'] != references[r['id']] for r in rows):
            raise ValueError('Original and processed data disagree')
    return splits


class PairCollator:
    def __init__(self, processor, expected_tokens, epoch=0):
        self.processor, self.expected_tokens, self.epoch = processor, expected_tokens, epoch
        self.tokenizer = processor.tokenizer
        self.image_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        # Identify actual first tokens inside the JSON value, without assuming
        # stand-alone word tokenization matches tokenization after a quote.
        candidates = []
        for state in STATES:
            s = ' {"care":"'+state+'"}'
            features = self.tokenizer(s,add_special_tokens=False,return_offsets_mapping=True)
            start = s.index(state)
            indices = [i for i,(a,b) in enumerate(features['offset_mapping']) if a <= start < b]
            if len(indices) != 1:
                raise ValueError('Ambiguous state token boundary')
            candidates.append(features['input_ids'][indices[0]])
        if len(set(candidates)) != 3:
            raise ValueError('State first tokens are not distinct')
        self.candidates = candidates

    @lru_cache(maxsize=243)
    def answer_features(self, serialized):
        answer = ' '+serialized
        features = self.tokenizer(answer,add_special_tokens=False,return_offsets_mapping=True)
        positions, classes = [], []
        for key in FOUNDATION_KEYS:
            match = re.search('"'+key+'":"([^"]+)"', answer)
            start, state = match.start(1), match.group(1)
            indices = [i for i,(a,b) in enumerate(features['offset_mapping']) if a <= start < b]
            if len(indices) != 1:
                raise ValueError('Ambiguous answer token boundary')
            pos, cls = indices[0], STATES.index(state)
            if features['input_ids'][pos] != self.candidates[cls]:
                raise ValueError('State token differs from candidate vocabulary')
            positions.append(pos); classes.append(cls)
        return features['input_ids'], positions, classes

    def __call__(self, rows):
        import torch
        from PIL import Image, ImageOps
        images = []
        for r in rows:
            with Image.open(r['image']) as im:
                images.append(ImageOps.exif_transpose(im).convert('RGB'))
        visual = self.processor(text=[image_prompt()]*len(rows),images=images,padding=True,
                                truncation=False,return_tensors='pt')
        captions = [r['captions'][self.epoch % len(r['captions'])] for r in rows]
        text = self.tokenizer([text_prompt(c) for c in captions],padding=True,truncation=False,return_tensors='pt')
        result = {}
        targets = [json.dumps(r['target'],separators=(',',':')) for r in rows]
        for modality, inputs in [('image',visual),('text',text)]:
            examples, state_positions, state_classes = [], [], []
            for i, serialized in enumerate(targets):
                prompt = unpad_input_ids(inputs['input_ids'][i].tolist(),inputs['attention_mask'][i].tolist())
                if prompt.count(self.image_id) != (self.expected_tokens if modality == 'image' else 0):
                    raise ValueError('Wrong image-token count')
                answer, positions, classes = self.answer_features(serialized)
                complete = answer+[self.tokenizer.eos_token_id]
                if len(prompt)+len(complete)>2048:
                    raise ValueError('Context overflow; no truncation')
                examples.append({'input_ids':prompt+complete,'attention_mask':[1]*(len(prompt)+len(complete)),
                                 'labels':[-100]*len(prompt)+complete})
                # Causal logits at t-1 predict the state token at t.
                state_positions.append([len(prompt)+p-1 for p in positions])
                state_classes.append(classes)
            batch = pad_supervised_batch(examples,pad_token_id=self.tokenizer.pad_token_id)
            result[modality] = {k:torch.tensor(v,dtype=torch.long) for k,v in batch.items()}
            if modality == 'image':
                result[modality]['pixel_values'] = visual['pixel_values']
            result[modality+'_positions'] = torch.tensor(state_positions)
            result['state_classes'] = torch.tensor(state_classes)
        result['candidate_ids'] = torch.tensor(self.candidates)
        return result


def state_logits(logits, positions, candidates):
    import torch
    rows = torch.arange(logits.shape[0],device=logits.device)[:,None]
    return logits[rows,positions.to(logits.device)].index_select(-1,candidates.to(logits.device)).float()


def paired_state_loss(image_logits, text_logits, classes):
    import torch
    from torch.nn import functional as F
    a, b = image_logits.float(), text_logits.float()
    supervised = .5*(F.cross_entropy(a.reshape(-1,3),classes.reshape(-1))+
                      F.cross_entropy(b.reshape(-1,3),classes.reshape(-1)))
    log_a, log_b = F.log_softmax(a,-1), F.log_softmax(b,-1)
    pa, pb = log_a.exp(), log_b.exp()
    mean = ((pa+pb)/2).clamp_min(1e-12)
    js = .5*((pa*(log_a-mean.log())).sum(-1)+(pb*(log_b-mean.log())).sum(-1)).mean()
    return supervised, js


def validation(model, processor, rows, batch_size, output):
    import torch
    from PIL import Image, ImageOps
    predictions = {}
    processor.tokenizer.padding_side = 'left'
    model.eval()
    trie = AnswerTrie(processor.tokenizer)
    with torch.inference_mode():
        for modality in ['image','text']:
            for start in range(0,len(rows),batch_size):
                batch = rows[start:start+batch_size]
                if modality == 'image':
                    images=[]
                    for r in batch:
                        with Image.open(r['image']) as im:images.append(ImageOps.exif_transpose(im).convert('RGB'))
                    inputs=processor(text=[image_prompt()]*len(batch),images=images,padding=True,truncation=False,return_tensors='pt')
                else:
                    inputs=processor.tokenizer([text_prompt(r['captions'][0]) for r in batch],padding=True,truncation=False,return_tensors='pt')
                width=inputs['input_ids'].shape[1]
                inputs={k:v.to('cuda',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k,v in inputs.items()}
                generated=model.generate(**inputs,max_new_tokens=256,do_sample=False,num_beams=1,use_cache=True,
                    prefix_allowed_tokens_fn=lambda batch_id,ids:trie.allowed(ids[width:].tolist()),
                    pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
                for r,tokens in zip(batch,generated[:,width:]):
                    raw=processor.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    predictions[r['id'],modality]={'raw_prediction':raw,'prediction':parse_prediction(raw)}
    correct_image=correct_text=consistent=joint=0
    records=[]
    for r in rows:
        a,b=[predictions[r['id'],m]['prediction'] for m in ['image','text']]
        consistent+=int(a==b);correct_image+=int(a==r['target']);correct_text+=int(b==r['target'])
        joint+=int(a==b==r['target'])
        records.append({'id':r['id'],'image':predictions[r['id'],'image'],'text':predictions[r['id'],'text'],
                        'reference':r['target'],'consistent':a==b})
    metrics={'num_scenes':len(rows),'strict_agreement':consistent/len(rows),
             'image_exact_match':correct_image/len(rows),'text_exact_match':correct_text/len(rows),
             'joint_correct_rate':joint/len(rows),'consistent_count':consistent}
    dump(output,{'metrics':metrics,'predictions':records})
    processor.tokenizer.padding_side='right'
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,default=ROOT/'runs/moral-llava-1.5')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--pair-batch-size',type=int,default=4)
    parser.add_argument('--learning-rate',type=float,default=2e-5)
    parser.add_argument('--state-weight',type=float,default=.5)
    parser.add_argument('--consistency-weight',type=float,default=1.)
    parser.add_argument('--quality-tolerance',type=float,default=.02)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.epochs<1 or args.pair_batch_size<1 or args.learning_rate<=0 or min(args.state_weight,args.consistency_weight,args.quality_tolerance)<0:
        raise ValueError('Invalid hyperparameters')
    import torch
    import peft,transformers
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor,LlavaForConditionalGeneration
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8)
    started=time.monotonic()
    splits=load_splits(ROOT/'data/raw/SMID_moralclip')
    train_rows,validation_rows=splits['train'],splits['validation']
    if args.smoke:train_rows=train_rows[:8];validation_rows=validation_rows[:8]
    args.output.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update({'started_at':datetime.now(timezone.utc).isoformat(),'seed':42,'base_revision':REVISION,
        'train_scenes':len(train_rows),'validation_scenes':len(validation_rows),'test_access':False,
        'selection_rule':'max validation strict agreement; mean image/text exact and joint correct no worse than initialization minus tolerance',
        'caption_policy':'training index (epoch-1)%10; validation fixed caption 0','objective':'.5*(image_LM_CE+text_LM_CE)+state_weight*state_CE+consistency_weight*state_JS',
        'state_JS_scope':'three first-token probabilities at five teacher-forced answer value positions; not sequence-level agreement',
        'source_sha256':sha256(Path(__file__)),'manifest_sha256':sha256(ROOT/'data/raw/SMID_moralclip/smid_manifest.json'),
        'initial_adapter_sha256':sha256(args.adapter/'adapter_model.safetensors'),
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__}})
    dump(args.output/'run_config.json',config)
    status={'status':'loading','epoch':0,'step':0,'test_evaluated':False}
    dump(args.output/'status.json',status)
    processor=AutoProcessor.from_pretrained(str(BASE),local_files_only=True)
    base=LlavaForConditionalGeneration.from_pretrained(str(BASE),torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda')
    expected=configure_processor(processor,base.config,padding_side='right')
    for p in base.parameters():p.requires_grad_(False)
    model=peft.PeftModel.from_pretrained(base,str(args.adapter),is_trainable=True)
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n in trainable):raise RuntimeError('Freeze policy violated')
    config['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    config['trainable_names']=trainable
    dump(args.output/'run_config.json',config)
    baseline=validation(model,processor,validation_rows,8,args.output/'validation_initial.json')
    best_metric,best_epoch=baseline['strict_agreement'],0
    print(json.dumps({'initial_validation':baseline}),flush=True)
    collator=PairCollator(processor,expected)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=.01)
    steps_per_epoch=math.ceil(len(train_rows)/args.pair_batch_size)
    total_steps=steps_per_epoch*args.epochs
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:.5*(1+math.cos(math.pi*min(1.,step/max(total_steps,1)))))
    step=0
    for epoch in range(1,args.epochs+1):
        collator.epoch=epoch-1
        loader=DataLoader(train_rows,batch_size=args.pair_batch_size,shuffle=True,collate_fn=collator,
                          num_workers=0,generator=torch.Generator().manual_seed(42+epoch))
        model.train();model.config.use_cache=False
        for batch in loader:
            t=time.monotonic();optimizer.zero_grad(set_to_none=True)
            outputs=[];states=[]
            for modality in ['image','text']:
                inputs={k:v.to('cuda',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k,v in batch[modality].items()}
                output=model(**inputs,use_cache=False)
                outputs.append(output)
                states.append(state_logits(output.logits,batch[modality+'_positions'],batch['candidate_ids']))
            state_ce,js=paired_state_loss(states[0],states[1],batch['state_classes'].to('cuda'))
            lm=.5*(outputs[0].loss+outputs[1].loss)
            loss=lm+args.state_weight*state_ce+args.consistency_weight*js
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1
            event={'epoch':epoch,'step':step,'loss':float(loss.detach()),'lm_loss':float(lm.detach()),
                   'state_ce':float(state_ce.detach()),'state_js':float(js.detach()),'grad_norm':float(norm),
                   'seconds':time.monotonic()-t,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
            with (args.output/'train_log.jsonl').open('a') as handle:handle.write(json.dumps(event)+'\n')
            if step==1 or step%25==0:print(json.dumps(event),flush=True)
            status.update({'status':'training','epoch':epoch,'step':step,'last_loss':event['loss']})
            if step%25==0:dump(args.output/'status.json',status)
            del outputs,states,output,loss,lm,state_ce,js
            if args.smoke and step>=2:break
        metrics=validation(model,processor,validation_rows,8,args.output/f'validation_epoch_{epoch:03d}.json')
        baseline_mean=(baseline['image_exact_match']+baseline['text_exact_match'])/2
        mean=(metrics['image_exact_match']+metrics['text_exact_match'])/2
        eligible=mean>=baseline_mean-args.quality_tolerance and metrics['joint_correct_rate']>=baseline['joint_correct_rate']-args.quality_tolerance
        checkpoint=args.output/f'epoch-{epoch:03d}'
        model.save_pretrained(checkpoint,safe_serialization=True,save_embedding_layers=False)
        processor.save_pretrained(checkpoint)
        meta={'format_version':1,'base_model_id':'llava-hf/llava-1.5-7b-hf','resolved_base_commit':REVISION,
              'task':'paired_image_text_moral_sft_state_js','initial_adapter':str(args.adapter),
              'epoch':epoch,'step':step,'validation':metrics,'quality_eligible':eligible,'training_config':config}
        dump(checkpoint/'training_metadata.json',meta)
        if eligible and metrics['strict_agreement']>best_metric:
            best_metric,best_epoch=metrics['strict_agreement'],epoch
        dump(args.output/'selection.json',{'selected_epoch':best_epoch,'selected_adapter':str(args.output/f'epoch-{best_epoch:03d}') if best_epoch else str(args.adapter),
             'best_validation_agreement':best_metric,'selection_split':'validation','quality_tolerance':args.quality_tolerance})
        print(json.dumps({'epoch':epoch,'validation':metrics,'quality_eligible':eligible,'selected_epoch':best_epoch}),flush=True)
        if args.smoke:break
    status.update({'status':'smoke_complete' if args.smoke else 'complete','step':step,'selected_epoch':best_epoch,
                   'total_seconds':time.monotonic()-started,'finished_at':datetime.now(timezone.utc).isoformat()})
    dump(args.output/'status.json',status)


if __name__=='__main__':main()
