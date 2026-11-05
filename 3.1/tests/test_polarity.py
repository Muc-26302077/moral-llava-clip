"""Protocol, leakage, unequal-support TICR and real tiny-model supervision checks."""
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from moral_llava import polarity
from moral_llava.schema import LABEL_PAIRS


def source(identifier, labels, split='test'):
    return {'id':identifier, 'split':split, 'image_path':f'{split}/{identifier}.png',
            'moral_labels':labels, 'captions':['A factual scene description.']*10}


def fixture():
    rows = []
    # Care has five times the support of each other task: macro != micro.
    for group in ('A','B'):
        for task in polarity.TASKS:
            for i in range(10 if task == 'care' else 2):
                label = LABEL_PAIRS[task][i%2]
                rows.extend(polarity.make_records(source(f'{group}-{i}', [label]), group))
    audit = audit_for(rows)
    predictions = []
    for row in rows:
        normalized = {k:row[k] for k in ('id','scene_id','task','modality','group','caption_index')}
        normalized['prompt'] = f'USER: {row["conversations"][0]["value"]} ASSISTANT:'
        predictions.append(polarity.format_prediction(normalized,row['conversations'][1]['value']))
    return rows, predictions, audit


def audit_for(rows):
    return {'protocol':polarity.VERSION, 'source_scene_counts':{'test':126},
            'test_groups_original_scenes':{'A':63,'B':63}, 'question_counts':{'test':len(rows)},
            'test_record_ids_sha256':polarity.record_ids_sha(rows),
            'test_eligible_scene_groups':{g:sorted({r['scene_id'] for r in rows if r['group']==g}) for g in ('A','B')},
            'test_class_counts_by_group':{g:{t:dict(Counter(r['conversations'][1]['value'] for r in rows
                if r['task']==t and r['group']==g and r['modality']=='image')) for t in polarity.TASKS} for g in ('A','B')}}


class PolarityTests(unittest.TestCase):
    def test_care_and_harm_are_opposite_and_unannotated_is_excluded(self):
        self.assertIs(polarity.targets(['Care'])['care'],True)
        self.assertIs(polarity.targets(['Harm'])['care'],False)
        self.assertIsNone(polarity.targets(['Harm'])['fairness'])
        self.assertEqual(polarity.make_records(source('neutral',['non-moral'])),[])
        harm = polarity.make_records(source('harm',['Harm']))
        self.assertEqual(len(harm),2)
        self.assertEqual({r['task'] for r in harm},{'care'})
        self.assertTrue(all(r['conversations'][1]['value']=='伤害' for r in harm))
        self.assertNotIn('是否涉及',harm[0]['conversations'][0]['value'])
        self.assertNotIn('A factual scene',harm[0]['conversations'][0]['value'])
        self.assertNotIn('<image>',harm[1]['conversations'][0]['value'])
        with self.assertRaises(ValueError):polarity.targets(['Care','Harm'])

    def test_labels_are_task_specific_and_strict(self):
        self.assertTrue(polarity.parse_answer(' 关怀\n','care'))
        self.assertFalse(polarity.parse_answer('伤害','care'))
        for raw in ('是','否','关怀。','Care','公平','伤害，因为',True,None):
            with self.subTest(raw=raw),self.assertRaises(ValueError):polarity.parse_answer(raw,'care')

    def test_perfect_unequal_support_benchmark_and_audit_required(self):
        rows,predictions,audit=fixture()
        result=polarity.score(rows,predictions,audit)
        self.assertTrue(result['complete_grouped_benchmark'])
        self.assertTrue(result['eligible_for_stage2'])
        self.assertEqual(result['overall']['primary_macro']['ticr'],1)
        self.assertEqual(result['metric_name'],'TICR')
        self.assertFalse(polarity.score(rows,predictions)['complete_grouped_benchmark'])
        partial=[r for r in rows if r['task']!='fairness']
        pred=[r for r in predictions if r['task']!='fairness']
        self.assertFalse(polarity.score(partial,pred,audit)['complete_grouped_benchmark'])

    def test_macro_threshold_does_not_use_pooled_pair_count(self):
        rows,predictions,audit=fixture()
        for row in predictions:
            if row['modality']=='text' and (row['task'] in ('fairness','loyalty') or
                    (row['task']=='authority' and row['scene_id'].endswith('-0'))):
                answer=polarity.parse_answer(row['raw_prediction'],row['task'])
                row['raw_prediction']=polarity.ANSWERS[row['task']][not answer]
        result=polarity.score(rows,predictions,audit)
        macro=result['groups']['A']['primary_macro']
        self.assertAlmostEqual(macro['ticr'],.5)
        self.assertGreater(macro['micro_ticr'],.6)
        self.assertFalse(result['ticr_passes_60pct_both_groups'])

    def test_constant_agreement_fails_quality_and_invalid_never_agrees(self):
        rows,predictions,audit=fixture()
        for row in predictions:row['raw_prediction']=polarity.ANSWERS[row['task']][True]
        result=polarity.score(rows,predictions,audit)
        self.assertEqual(result['overall']['primary_macro']['ticr'],1)
        self.assertFalse(result['above_chance_quality_guard'])
        for row in predictions:
            row['raw_prediction']='是';row['prediction']='关怀'
        result=polarity.score(rows,predictions,audit)
        self.assertEqual(result['overall']['primary_macro']['ticr'],0)
        self.assertEqual(result['overall']['primary_macro']['joint_accuracy'],0)

    def test_missing_duplicate_changed_prompt_and_provenance_rejected(self):
        rows,pred,audit=fixture()
        for bad in (pred[:-1],pred+[pred[0]]):
            with self.assertRaises(ValueError):polarity.score(rows,bad,audit)
        for key,value in (('prompt_sha256','changed'),('smoke_only',True),('adapter_sha256','other'),('caption_index',7)):
            bad=copy.deepcopy(pred);bad[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):polarity.score(rows,bad,audit)

    def test_inference_never_reads_reference_answer(self):
        class Guard(dict):
            def __getitem__(self,key):raise AssertionError('Read assistant answer')
            def get(self,*args,**kwargs):raise AssertionError('Read assistant answer')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'test').mkdir();(root/'test/scene.png').touch()
            rows=polarity.make_records(source('scene',['Care']))
            for row in rows:row['conversations'][1]=Guard()
            with patch.object(polarity,'read',return_value=rows):
                result=polarity.load_records(root/'unused.json',root,False)
            self.assertTrue(all(r['target_json'] is None for r in result))
            self.assertIsNone(result[1]['image_path'])

    def test_prepare_keeps_source_splits_and_excludes_neutral_per_task(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir()
            sources=[source('train-care',['Care'],'train'),source('val-harm',['Harm'],'validation'),
                     source('test-both',['Care','Cheating']),source('test-neutral',['non-moral'])]
            for i,row in enumerate(sources):
                path=raw/row['image_path'];path.parent.mkdir(exist_ok=True);path.write_bytes(bytes([i]))
            (raw/'smid_manifest.json').write_text(json.dumps(sources[:2]))
            (raw/'test/smid_manifest.json').write_text(json.dumps(sources[2:]))
            groups=root/'groups.json';groups.write_text(json.dumps({'scenes':[
                {'id':'test-both','group':'A'},{'id':'test-neutral','group':'B'}]}))
            audit=polarity.prepare(raw,root/'output',groups)
            self.assertEqual(audit['source_scene_counts'],{'train':1,'validation':1,'test':2})
            self.assertEqual(audit['question_counts'],{'train':2,'validation':2,'test':4})
            test=polarity.read(root/'output/test.json')
            self.assertEqual({r['task'] for r in test},{'care','fairness'})
            self.assertEqual(audit['no_polarity_scene_ids']['test'],['test-neutral'])
            with self.assertRaises(FileExistsError):polarity.prepare(raw,root/'output',groups)

    def test_majority_is_fitted_only_on_training_and_keeps_polarity_labels(self):
        rows,_,audit=fixture()
        train=copy.deepcopy(rows)
        for row in train:row['split']='train'
        pred=polarity.majority_predictions(train,rows)
        self.assertEqual(polarity.score(rows,pred,audit)['overall']['primary_macro']['ticr'],1)
        with self.assertRaises(ValueError):polarity.majority_predictions(rows,rows)

    def test_entrypoint_help_is_portable(self):
        entry=Path(polarity.__file__).resolve().parents[1]/'run_polarity.py'
        with tempfile.TemporaryDirectory() as folder:
            for command in ('prepare','train','infer','score','majority'):
                result=subprocess.run([sys.executable,'-B',str(entry),command,'--help'],cwd=folder,
                                      capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)


class TinyPolarityModelTests(unittest.TestCase):
    def test_real_mixed_modality_loss_masks_and_language_only_gradients(self):
        import torch
        from PIL import Image
        from peft import LoraConfig,TaskType,get_peft_model
        from test_alignment_model import AlignmentModelTests
        from moral_llava.train import configure_processor,find_language_lora_targets
        from moral_llava.polarity_runtime import PolaritySFTCollator
        torch.set_num_threads(1);torch.manual_seed(42)
        processor=AlignmentModelTests.tiny_processor()
        processor.tokenizer.add_tokens(['关怀','伤害'])
        model=AlignmentModelTests.tiny_llava();model.resize_token_embeddings(len(processor.tokenizer))
        expected=configure_processor(processor,model.config,padding_side='right')
        for parameter in model.parameters():parameter.requires_grad_(False)
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=2,lora_alpha=4,
            target_modules=find_language_lora_targets(model,torch.nn.Linear)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'image.png';Image.new('RGB',(8,8),'red').save(path)
            rows=[{'task':'care','modality':'image','prompt':'<image> word4','image_path':str(path),'target_json':'关怀'},
                  {'task':'care','modality':'text','prompt':'word5 word6','image_path':None,'target_json':'伤害'}]
            collator=PolaritySFTCollator(processor,image_token_id=127,expected_image_tokens=expected,max_length=128)
            batch=collator(rows)
            self.assertEqual(tuple(batch['pixel_values'].shape),(1,3,8,8))
            self.assertEqual(int((batch['input_ids'][1]==127).sum()),0)
            loss=model(**batch).loss;self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_' in n))
            self.assertTrue(all(p.grad is None for p in model.vision_tower.parameters()))
            self.assertTrue(all(p.grad is None for p in model.multi_modal_projector.parameters()))
            for row in batch['labels']:
                active=(row!=-100).nonzero().flatten()
                self.assertEqual(int(row[active[-1]]),processor.tokenizer.eos_token_id)
                self.assertTrue((row[:active[0]]==-100).all())
            text=collator(rows[1:]);self.assertNotIn('pixel_values',text)
            self.assertTrue(torch.isfinite(model(**text).loss))
            rows[0]['target_json']='是'
            with self.assertRaises(ValueError):collator(rows)


if __name__=='__main__':
    unittest.main()
