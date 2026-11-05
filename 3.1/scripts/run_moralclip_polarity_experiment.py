"""Two-stage original-MoralCLIP integration baseline; run in detached screen."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moral_llava import polarity
from moral_llava.moralclip_integration import MODEL_ID,REVISION,WEIGHT_SHA256,sha,resolve_source
from run_polarity_experiment import independent_audit


def now():return datetime.now(timezone.utc).isoformat()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--moralclip-model',default=MODEL_ID)
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if args.resume:
        state=polarity.read(output/'status.json')
        for key in ('runner_pid','child_pid'):
            pid=state.get(key)
            path=Path(f'/proc/{pid}/cmdline')
            if pid!=os.getpid() and path.exists() and b'moralclip_polarity' in path.read_bytes():
                raise RuntimeError('Experiment is still live; refusing duplicate launch')
    else:
        output.mkdir(parents=True,exist_ok=False)
        state={'status':'starting','started_at':now(),'commands':[]}
    state.update(runner_pid=os.getpid(),child_pid=None,output=str(output))
    environment=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    if args.local_files_only:environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    offline=['--local-files-only'] if args.local_files_only else []
    entry=str(ROOT/'run_moralclip_polarity.py')

    def record():
        state['heartbeat_at']=now();save(output/'status.json',state)

    def execute(name,arguments,products=(),steps=None):
        previous=next((c for c in reversed(state['commands']) if c['name']==name and c.get('returncode')==0),None)
        if previous:
            for path,checksum in previous.get('products',{}).items():
                if not Path(path).is_file() or sha(path)!=checksum:
                    raise RuntimeError(f'Completed stage changed: {name}: {path}')
            print('SKIP verified '+name,flush=True);return
        logfile=output/f'{name}.log'
        attempt={'name':name,'argv':[sys.executable,'-u']+arguments,'started_at':now(),'log':str(logfile)}
        state['commands'].append(attempt);state.update(status='running',phase=name)
        print(f'{now()} START {name}',flush=True);started=time.monotonic()
        with logfile.open('a') as log:
            child=subprocess.Popen(attempt['argv'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;record()
            while child.poll() is None:
                time.sleep(10)
                if steps:
                    with logfile.open('rb') as stream:
                        stream.seek(max(0,logfile.stat().st_size-32768));tail=stream.read().decode(errors='replace')
                    matches=re.findall(r'(\d+)/'+str(steps)+r' \[',tail)
                    if matches:state['training_progress']={'stage':name,'step':int(matches[-1]),'total':steps}
                record()
        attempt.update(returncode=child.returncode,finished_at=now(),seconds=round(time.monotonic()-started,3))
        state['child_pid']=None
        if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}; inspect {logfile}')
        attempt['products']={str(p):sha(p) for p in products};record()
        print(f'{now()} DONE {name}',flush=True)

    try:
        data=ROOT/'data/polarity-v1'
        if not data.is_dir():execute('prepare',[entry,'prepare'],[data/'audit.json'])
        audit=polarity.read(data/'audit.json')
        rows={s:polarity.read(data/f'{s}.json') for s in ('train','validation','test')}
        if audit['source_scene_counts']!={'train':2138,'validation':134,'test':126} or audit['question_counts']!={'train':9806,'validation':580,'test':576}:
            raise ValueError('Frozen original data coverage changed')
        for split,records in rows.items():
            if len(records)!=audit['question_counts'][split] or sha(data/f'{split}.json')!=audit['split_record_sha256'][split]:
                raise ValueError(f'Polarity data audit mismatch: {split}')
        scene_sets=[{r['scene_id'] for r in records} for records in rows.values()]
        if any(scene_sets[i]&scene_sets[j] for i,j in ((0,1),(0,2),(1,2))):raise ValueError('Cross-split scene leakage')
        source=resolve_source(args.moralclip_model,REVISION,args.local_files_only)
        if sha(source/'model.safetensors')!=WEIGHT_SHA256:raise ValueError('Not the user-confirmed original MoralCLIP')
        vision_config=polarity.read(source/'config.json')
        if vision_config.get('model_type')!='clip' or vision_config['vision_config'].get('patch_size')!=16:
            raise ValueError('Original MoralCLIP must be CLIP ViT-B/16')
        # Verify the language base snapshot before any model training.
        from moral_llava.pipeline import resolve_base
        base,base_checks=resolve_base(offline=args.local_files_only)
        save(output/'base_model.json',{'snapshot':str(base),'files':base_checks})
        sources=[ROOT/'run_moralclip_polarity.py',Path(__file__).resolve(),ROOT/'scripts/run_polarity_experiment.py']+sorted((ROOT/'moral_llava').glob('*.py'))
        config={'method':'original_moralclip_vision_polarity_sft_v1','protocol':polarity.VERSION,
                'base_model':'llava-hf/llava-1.5-7b-hf','base_revision':'b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
                'moralclip_model':str(args.moralclip_model),'moralclip_revision':REVISION,'moralclip_sha256':WEIGHT_SHA256,
                'vision_frozen':True,'feature_layer':-2,'image_tokens':196,'projector':[768,4096,4096],
                'warmup':{'epochs':1,'image_only':True,'examples':4903,'validation_examples':290,
                          'projector_learning_rate':1e-3,'language_lora_frozen':True,'steps':613},
                'joint':{'epochs':3,'examples':9806,'validation_examples':580,
                         'language_learning_rate':2e-5,'projector_learning_rate':1e-4,'steps':3678},
                'batch_size':8,'gradient_accumulation_steps':1,'precision':'bf16','gradient_checkpointing':False,
                'lora_r':16,'lora_alpha':32,'lora_dropout':.05,'seed':42,'scheduler':'cosine','warmup_ratio':.03,
                'weight_decay':0.,'optimizer':'AdamW betas=(0.9,0.999) eps=1e-8','max_length':2048,
                'selection':'minimum validation answer loss in each stage; no test-based selection',
                'initial_language_adapter':'fresh LoRA; no historical SFT adapter',
                'max_new_tokens':16,'decoding':'free greedy; no forced labels or agreement',
                'data_sha256':{s:sha(data/f'{s}.json') for s in rows},'audit_sha256':sha(data/'audit.json'),
                'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
                'test_policy':'historically reused fixed test; caption0; image and text paths independent',
                'threshold':'A/B macro TICR each >=60%; image/text macro balanced accuracy each >50%',
                'scope':'integration baseline only; no vision fine-tuning or text feedback'}
        frozen=output/'frozen_config.json'
        if frozen.exists():
            if polarity.read(frozen)!=config:raise ValueError('Frozen config, code or data changed')
        else:save(frozen,config)
        state['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();record()
        derived=output/'derived';derived.mkdir(exist_ok=True)
        for split in ('train','validation'):
            image_rows=[r for r in rows[split] if r['modality']=='image']
            save(derived/f'{split}-images.json',image_rows)
            scene_ids=set(r['scene_id'] for r in rows[split][:48])
            smoke=[r for r in rows[split] if r['scene_id'] in scene_ids]
            save(derived/f'{split}-smoke.json',smoke)
            save(derived/f'{split}-smoke-images.json',[r for r in smoke if r['modality']=='image'])
        training=[entry,'train','--moralclip-model',str(args.moralclip_model),'--batch-size','8','--eval-batch-size','8',
                  '--gradient-accumulation-steps','1','--seed','42','--precision','bf16','--no-gradient-checkpointing',
                  '--logging-steps','25']+offline
        warm=training+['--moralclip-stage','projector','--epochs','1','--learning-rate','1e-3','--projector-learning-rate','1e-3',
                      '--train-data',str(derived/'train-images.json'),'--validation-data',str(derived/'validation-images.json')]
        joint=training+['--moralclip-stage','joint','--epochs','3','--learning-rate','2e-5','--projector-learning-rate','1e-4']
        def continue_training(name,arguments,destination,steps):
            if args.resume and destination.exists() and not any(c['name']==name and c.get('returncode')==0 for c in state['commands']):
                checkpoints=sorted(destination.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
                if checkpoints:arguments+=['--resume-from-checkpoint',str(checkpoints[-1])]
                else:raise RuntimeError(f'Incomplete {name} without checkpoint; inspect before restarting')
            execute(name,arguments+['--output-dir',str(destination)],
                    [destination/'adapter_model.safetensors',destination/'training_metadata.json'],steps)
        continue_training('smoke-projector',warm+['--train-data',str(derived/'train-smoke-images.json'),
                          '--validation-data',str(derived/'validation-smoke-images.json'),'--max-steps','2','--logging-steps','1'],output/'smoke-projector',2)
        continue_training('smoke-joint',joint+['--train-data',str(derived/'train-smoke.json'),
                          '--validation-data',str(derived/'validation-smoke.json'),'--max-steps','2','--logging-steps','1',
                          '--initial-adapter',str(output/'smoke-projector')],output/'smoke-joint',2)
        inference=[entry,'infer','--device','cuda','--precision','bf16','--batch-size','8',
                   '--moralclip-model',str(args.moralclip_model)]+offline
        execute('smoke-reload',inference+['--adapter',str(output/'smoke-joint'),'--data',str(derived/'validation-smoke.json'),
                '--limit','16','--output',str(output/'smoke-reload.jsonl')],[output/'smoke-reload.jsonl'])
        continue_training('projector-warmup',warm,output/'projector',613)
        continue_training('polarity-train',joint+['--initial-adapter',str(output/'projector')],output/'adapter',3678)
        # Both test evaluations happen only after training/selection is fixed.
        execute('majority',[entry,'majority','--output',str(output/'majority.jsonl')],[output/'majority.jsonl'])
        for method,adapter in (('projector_only',output/'projector'),('trained',output/'adapter')):
            execute(method+'-infer',inference+['--adapter',str(adapter),'--output',str(output/f'{method}.jsonl')],[output/f'{method}.jsonl'])
        summary={'method':config['method'],'frozen_config_sha256':sha(frozen),'methods':{},'training':{}}
        for stage,directory,steps in (('projector',output/'projector',613),('joint',output/'adapter',3678)):
            meta=polarity.read(directory/'training_metadata.json')
            if meta['global_step']!=steps or not math.isfinite(meta['best_validation_loss']):raise ValueError('Training incomplete')
            if not meta.get('frozen_vision_verified_after_training'):raise ValueError('Frozen vision was not verified')
            if sha(Path(meta['best_model_checkpoint'])/'adapter_model.safetensors')!=sha(directory/'adapter_model.safetensors'):
                raise ValueError('Final checkpoint differs from validation-selected projector/LoRA')
            summary['training'][stage]={'steps':steps,'best_validation_loss':meta['best_validation_loss'],
                    'best_model_checkpoint':meta['best_model_checkpoint'],'adapter_sha256':sha(directory/'adapter_model.safetensors'),
                    'moralclip_integration':meta['moralclip_integration'],'frozen_vision_verified':True}
        lines=['# 原始 MoralCLIP 接入 LLaVA：极性问答基线','',
               '原始 MoralCLIP ViT-B/16 替换视觉塔；逐张量核验全部视觉权重。224×224 输入、196 patch token、倒数第二层特征，投影器为 768→4096→4096。',
               '第一阶段仅训练新投影器（4903 条图像问答，1 epoch，LR 1e-3）；第二阶段训练投影器（LR 1e-4）与全新语言 LoRA（LR 2e-5），9806 条图文独立问答、3 epochs。',
               '两阶段均冻结原始 MoralCLIP 及语言基座；batch 8、梯度累积 1、BF16、seed 42，按验证 loss 选模。投影器随 PEFT adapter 一起保存和重载。','',
               '| 方法 | 图像准确率（宏） | 文本准确率（宏） | TICR（宏） | 共同正确率（宏） | 图像 BA（宏） | 文本 BA（宏） |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for method in ('majority','projector_only','trained'):
            execute(method+'-score',[entry,'score','--predictions',str(output/f'{method}.jsonl'),
                    '--output',str(output/f'{method}-metrics.json')],[output/f'{method}-metrics.json'])
            metrics=polarity.read(output/f'{method}-metrics.json')
            predictions=[json.loads(s) for s in (output/f'{method}.jsonl').read_text().splitlines() if s.strip()]
            if not metrics['complete_grouped_benchmark']:raise ValueError('Incomplete test coverage')
            if method!='majority' and any(p.get('vision_weight_sha256')!=WEIGHT_SHA256 for p in predictions):
                raise ValueError('Wrong inference vision provenance')
            intervals=independent_audit(rows['test'],predictions,metrics)
            summary['methods'][method]={'metrics':metrics,'scene_bootstrap_ticr95':intervals}
            macro=metrics['overall']['primary_macro']
            fields=('image_accuracy','text_accuracy','ticr','joint_accuracy','image_balanced_accuracy','text_balanced_accuracy')
            lines.append('| '+method+' | '+' | '.join(f'{macro[k]:.2%}' for k in fields)+' |')
        trained=summary['methods']['trained']['metrics']
        lines+=['',f'A 组 TICR：{trained["groups"]["A"]["primary_macro"]["ticr"]:.2%}；B 组：{trained["groups"]["B"]["primary_macro"]["ticr"]:.2%}。',
                f'两组 ≥60%：{trained["ticr_passes_60pct_both_groups"]}；两路 BA >50%：{trained["above_chance_quality_guard"]}。',
                '测试为原 126 场景中符合极性标注的 87 场景、288 场景—维度对、576 条问答；各维度 72/42/63/55/56 对。按场景 bootstrap 2000 次的区间见 summary.json。',
                '本结果来自真实 MoralCLIP 视觉权重。历史原 LLaVA 视觉塔结果属于另一模型及训练初始化，不能据此单独归因视觉塔优劣。',
                'TICR 包含共同错误，需与准确率、平衡准确率和共同正确率同时解读。固定测试集已用于历史实验，并非全新盲测；上游 MoralCLIP 训练集与测试集重叠尚未核实。',
                '本次冻结视觉塔，未进行视觉微调或生成文本反馈视觉编码；原始 MoralCLIP 双向检索 R@1 基准不变。']
        for name,checksum in config['source_sha256'].items():
            if sha(ROOT/name)!=checksum:raise ValueError('Experiment code changed during run: '+name)
        save(output/'summary.json',summary);(output/'report.md').write_text('\n'.join(lines)+'\n')
        state.update(status='complete',phase='complete',finished_at=now(),report=str(output/'report.md'));record()
        print(f'{now()} COMPLETE {output}',flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error),finished_at=now());record();traceback.print_exc();raise


if __name__=='__main__':main()
