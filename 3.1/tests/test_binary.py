"""Small CPU checks only: no pretrained weights or GPU inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

from moral_llava import binary


def source(i=0, split='test', labels=None):
    return {'id': f's{i}', 'split': split, 'image_path': f'{split}/s{i}.png',
            'moral_labels': labels or (['Care','Loyalty'] if i % 2 else ['Fairness','Respect','Sanctity']),
            'captions': ['A factual scene description.']*10}


def fixture(n=126):
    rows = [r for i in range(n) for r in binary.make_records(source(i), 'A' if i < n//2 else 'B')]
    predictions = []
    for row in rows:
        normalized = {k: row[k] for k in ('id','scene_id','task','modality','group','caption_index')}
        normalized['prompt'] = f'USER: {row["conversations"][0]["value"]} ASSISTANT:'
        predictions.append(binary.format_prediction(normalized,row['conversations'][1]['value']))
    return rows, predictions


class BinaryTests(unittest.TestCase):
    def test_both_poles_mean_present_without_claiming_care_virtue(self):
        self.assertEqual(binary.targets(['Care']),binary.targets(['Harm']))
        self.assertTrue(binary.targets(['Harm'])['care'])
        self.assertEqual(binary.targets(['non-moral']),dict.fromkeys(binary.FOUNDATION_KEYS,False)|{'non_moral':True})

    def test_image_prompt_has_no_caption_and_text_has_no_image_token(self):
        rows=binary.make_records(source(labels=['Harm']))
        self.assertEqual(len(rows),12)
        for row in rows:
            prompt=row['conversations'][0]['value']
            if row['modality']=='image':
                self.assertEqual(prompt.count('<image>'),1)
                self.assertNotIn('A factual scene',prompt)
            else:
                self.assertNotIn('<image>',prompt)
                self.assertIn('A factual scene',prompt)

    def test_strict_answer_parser(self):
        self.assertTrue(binary.parse_answer(' 是\n'))
        self.assertFalse(binary.parse_answer('否'))
        for raw in ('是。','Yes','否，因为没有',True,None,'是\n否'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):binary.parse_answer(raw)

    def test_fully_correct_grouped_fixture(self):
        rows,pred=fixture()
        report=binary.score(rows,pred)
        self.assertTrue(report['eligible_for_stage2'])
        self.assertEqual(report['overall']['five_presence_joint_exact'],1)
        self.assertEqual(report['overall']['primary_macro']['agreement'],1)

    def test_constant_100pct_agreement_does_not_pass_quality(self):
        rows,pred=fixture()
        for row in pred:row['raw_prediction']='否'
        report=binary.score(rows,pred)
        self.assertTrue(report['agreement_passes_60pct_both_groups'])
        self.assertFalse(report['above_chance_quality_guard'])
        self.assertFalse(report['eligible_for_stage2'])

    def test_invalid_outputs_fail_and_cached_fields_are_ignored(self):
        rows,pred=fixture()
        for row in pred:
            row['raw_prediction']='Yes'
            row['prediction']=True
        report=binary.score(rows,pred)
        self.assertEqual(report['overall']['primary_macro']['agreement'],0)
        self.assertEqual(report['overall']['primary_macro']['joint_accuracy'],0)

    def test_missing_duplicate_changed_prompt_and_smoke_rejected(self):
        rows,pred=fixture(4)
        for bad in (pred[:-1],pred+[pred[0]]):
            with self.assertRaises(ValueError):binary.score(rows,bad)
        for key,value in (('prompt_sha256','wrong'),('smoke_only',True),('modality','wrong')):
            bad=copy.deepcopy(pred);bad[0][key]=value
            with self.assertRaises(ValueError):binary.score(rows,bad)
        bad=copy.deepcopy(pred);bad[0]['adapter_sha256']='different-model'
        with self.assertRaises(ValueError):binary.score(rows,bad)
        self.assertFalse(binary.score(rows,pred)['complete_grouped_benchmark'])

    def test_60pct_binary_threshold_is_189_of_315_per_group(self):
        rows,pred=fixture()
        # Disagree on exactly 126 questions in each group: 189/315 remain.
        for group in ('A','B'):
            changed=0
            for row in pred:
                if row['group']==group and row['modality']=='text' and row['task']!='non_moral' and changed<126:
                    row['raw_prediction']='否' if row['raw_prediction']=='是' else '是'
                    changed+=1
        self.assertTrue(binary.score(rows,pred)['agreement_passes_60pct_both_groups'])
        for row in pred:
            if row['group']=='A' and row['modality']=='text' and row['task']!='non_moral':
                truth=next(r['conversations'][1]['value'] for r in rows if r['id']==row['id'])
                if row['raw_prediction']==truth:
                    row['raw_prediction']='否' if truth=='是' else '是';break
        self.assertFalse(binary.score(rows,pred)['agreement_passes_60pct_both_groups'])

    def test_inference_does_not_read_assistant(self):
        class Guard(dict):
            def __getitem__(self,k):raise AssertionError('Read target during inference')
            def get(self,*a,**kw):raise AssertionError('Read target during inference')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'test').mkdir();(root/'test/s0.png').touch()
            rows=binary.make_records(source())
            for row in rows:row['conversations'][1]=Guard()
            with patch('moral_llava.binary.read',return_value=rows):
                result=binary.load_records(root/'fake.json',root,False)
            self.assertTrue(all(r['target_json'] is None for r in result))
            self.assertTrue(all(r['image_path'] is None for r in result if r['modality']=='text'))

    def test_conversion_keeps_splits_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw';raw.mkdir()
            records=[]
            for i,split in enumerate(('train','validation','test')):
                row=source(i,split);records.append(row)
                (raw/split).mkdir();(raw/row['image_path']).write_bytes(bytes([i]))
            (raw/'smid_manifest.json').write_text(json.dumps(records[:2]))
            (raw/'test/smid_manifest.json').write_text(json.dumps(records[2:]))
            groups=root/'groups.json';groups.write_text(json.dumps({'scenes':[{'id':'s2','group':'A'}]}))
            report=binary.prepare(raw,root/'binary',groups)
            self.assertEqual(report['scene_counts'],dict.fromkeys(('train','validation','test'),1))
            self.assertEqual(report['question_counts'],dict.fromkeys(('train','validation','test'),12))
            self.assertEqual(len(binary.read(root/'binary/tasks/care/train.json')),2)
            with self.assertRaises(FileExistsError):binary.prepare(raw,root/'binary',groups)

    def test_cross_split_exact_images_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'test').mkdir();(root/'train').mkdir();(root/'validation').mkdir()
            records=[source(i,s) for i,s in enumerate(('train','validation','test'))]
            for row in records:(root/row['image_path']).write_bytes(b'same bytes')
            (root/'smid_manifest.json').write_text(json.dumps(records[:2]))
            (root/'test/smid_manifest.json').write_text(json.dumps(records[2:]))
            with self.assertRaisesRegex(ValueError,'overlap'):binary.prepare(root,root/'out')
            self.assertFalse((root/'out').exists())

    def test_explicit_public_exclusions_do_not_copy_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'test').mkdir();(root/'train').mkdir();(root/'validation').mkdir()
            records=[source(i,s) for i,s in enumerate(('train','train','validation','test'))]
            for i,row in enumerate(records[1:],1):(root/row['image_path']).write_bytes(bytes([i]))
            (root/'smid_manifest.json').write_text(json.dumps(records[:3]))
            (root/'test/smid_manifest.json').write_text(json.dumps(records[3:]))
            exclude=root/'excluded.json';exclude.write_text(json.dumps({'images':[{'id':'s0','image_path':'train/s0.png'}]}))
            report=binary.prepare(root,root/'out',excluded_images=exclude)
            self.assertEqual(report['excluded_training_scene_ids'],['s0'])
            self.assertEqual(report['scene_counts']['train'],1)
            self.assertNotIn('s0',{r['scene_id'] for r in binary.read(root/'out/train.json')})

    def test_entrypoint_help_from_another_working_directory(self):
        entry=Path(binary.__file__).resolve().parents[1]/'run_binary.py'
        with tempfile.TemporaryDirectory() as directory:
            for command in ('prepare','train','infer','score','majority'):
                result=subprocess.run([sys.executable,'-B',str(entry),command,'--help'],cwd=directory,
                                      text=True,capture_output=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('usage:',result.stdout)

    def test_majority_fits_train_only(self):
        rows,_=fixture(4)
        train=[r for i in range(3) for r in binary.make_records(source(i,'train'))]
        pred=binary.majority_predictions(train,rows)
        self.assertEqual(binary.score(rows,pred)['overall']['primary_macro']['agreement'],1)
        with self.assertRaises(ValueError):binary.majority_predictions(rows,rows)


class TinyBinaryModelTests(unittest.TestCase):
    def test_mixed_modalities_real_processor_tiny_model_backward_and_text_only(self):
        # Only a random 2-layer model; never loads the pretrained 7B model.
        import torch
        from PIL import Image
        from test_alignment_model import AlignmentModelTests
        from moral_llava.binary_runtime import BinarySFTCollator
        from moral_llava.train import configure_processor,find_language_lora_targets
        from peft import LoraConfig,TaskType,get_peft_model
        torch.set_num_threads(1);torch.manual_seed(42)
        processor=AlignmentModelTests.tiny_processor()
        processor.tokenizer.add_tokens(['是','否'])
        model=AlignmentModelTests.tiny_llava()
        model.resize_token_embeddings(len(processor.tokenizer))
        expected=configure_processor(processor,model.config,padding_side='right')
        for parameter in model.parameters():parameter.requires_grad_(False)
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=2,lora_alpha=4,
            target_modules=find_language_lora_targets(model,torch.nn.Linear)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'image.png';Image.new('RGB',(8,8),'red').save(path)
            records=[{'modality':'image','prompt':'<image> word4','image_path':str(path),'target_json':'是'},
                     {'modality':'text','prompt':'word5 word6','image_path':None,'target_json':'否'}]
            collator=BinarySFTCollator(processor,image_token_id=127,expected_image_tokens=expected,max_length=128)
            batch=collator(records)
            self.assertEqual(tuple(batch['pixel_values'].shape),(1,3,8,8))
            self.assertEqual(int((batch['input_ids'][1]==127).sum()),0)
            loss=model(**batch).loss
            self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_' in n))
            self.assertTrue(all(p.grad is None for p in model.vision_tower.parameters()))
            text=collator(records[1:]);self.assertNotIn('pixel_values',text)
            self.assertTrue(torch.isfinite(model(**text).loss))
            for i,row in enumerate(batch['labels']):
                active=(row!=-100).nonzero().flatten()
                self.assertEqual(int(row[active[-1]]),processor.tokenizer.eos_token_id)
                self.assertTrue((row[:active[0]]==-100).all())


if __name__=='__main__':unittest.main()
