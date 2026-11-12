"""Scene guidance: independent deployment, paired supervision, and real CPU gradients."""
from argparse import Namespace
import copy
from pathlib import Path
import tempfile
import unittest

import torch
import tiny_model as fixture

from moral_llava.scene_guided_vision import (
    SceneConfig, SceneContext, caption_index, install_scene_guidance,
    paired_text_rows, save_scene_guidance, load_scene_guidance,
    answer_logits, supervised_and_distilled, INDEPENDENT, PAIRED,
)
import run_scene_guided_vision as runner


class SceneGuidedVisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def runtime(self):
        model = fixture.TinyModels.tiny_llava().eval()
        block, _ = install_scene_guidance(model, config=SceneConfig(vision_dim=24, text_dim=16, rank=4))
        return model, block

    @staticmethod
    def batch():
        ids = torch.tensor([[2, 127, 127, 127, 127, 4, 5, 3],
                            [2, 4, 5, 6, 7, 8, 5, 3]])
        labels = ids.clone()
        labels[:, :6] = -100
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                    pixel_values=torch.randn(1, 3, 8, 8))

    def test_zero_init_matches_baseline_and_freezes_original_model(self):
        model = fixture.TinyModels.tiny_llava().eval()
        batch = self.batch()
        expected = model(**batch).logits.detach()
        block, _ = install_scene_guidance(model, config=SceneConfig(vision_dim=24, text_dim=16, rank=4))
        for context in (SceneContext(), SceneContext('paired', torch.randn(1, 16))):
            with block.conditioned(context):
                torch.testing.assert_close(model(**batch).logits, expected, rtol=0, atol=0)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all('.adapter.' in n for n in trainable))
        with self.assertRaisesRegex(RuntimeError, 'explicit scene context'):
            model(**batch)

    def test_predicted_path_cannot_read_supervision_caption_but_paired_path_can(self):
        model, block = self.runtime()
        with torch.no_grad():
            block.adapter.residual.up.weight.normal_(std=.2)
        batch = self.batch()
        first, other = torch.randn(1, 16), torch.randn(1, 16)
        with block.conditioned(SceneContext('predicted', first)):
            a = model(**batch).logits.detach()
            align_a = block.last_alignment.detach()
        with block.conditioned(SceneContext('predicted', other)):
            b = model(**batch).logits.detach()
            align_b = block.last_alignment.detach()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertNotEqual(float(align_a), float(align_b))
        with block.conditioned(SceneContext('paired', first)):
            a = model(**batch).logits.detach()
        with block.conditioned(SceneContext('paired', other)):
            b = model(**batch).logits.detach()
        self.assertGreater(float((a[0]-b[0]).abs().max()), 1e-7)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)

    def test_actual_training_step_on_cpu_with_frozen_decoder_checkpointing(self):
        model, block = self.runtime()
        model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.language_model.model.training = True
        mixed = self.batch()
        image = {k: v[:1].clone() for k, v in mixed.items()}
        teacher_ids = torch.tensor([[2, 8, 9, 5, 3]])
        labels = teacher_ids.clone()
        labels[:, :3] = -100
        teacher = dict(input_ids=teacher_ids, attention_mask=torch.ones_like(teacher_ids), labels=labels)
        def collator(rows):
            return {k: v.clone() for k, v in (image if rows[0]['modality'] == 'image' else teacher).items()}
        text_feature = torch.randn(1, 16)
        class Encoder:
            def encode(self, captions):
                assert captions == ['Actual scene caption']
                return text_feature
        args = Namespace(temperature=2., distill_weight=.2, alignment_weight=.1,
                         preservation_weight=.05, paired_weight=.5, _batch_weight=1.)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW(block.adapter.parameters(), lr=.02, weight_decay=0)
        losses = []
        for _ in range(12):
            optimizer.zero_grad()
            report = runner.train_batch(args, model, None, block, torch.float32, Encoder(),
                {'scene': 'Actual scene caption'}, [{'scene_id': 'scene', 'modality': 'image'}],
                [{'modality': 'text'}], collator)
            optimizer.step()
            losses.append(report['independent_loss']+.5*report['paired_ce'])
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.adapter.predictor.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.adapter.residual.condition.parameters()))
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
        self.assertIsNone(block.last_alignment)
        self.assertFalse(block._active)

    def test_distillation_aligns_supervised_tokens_and_detaches_teacher(self):
        student = torch.randn(1, 5, 12, requires_grad=True)
        teacher = torch.randn(1, 7, 12, requires_grad=True)
        s_labels = torch.tensor([[-100, -100, -100, 4, 3]])
        t_labels = torch.tensor([[-100, -100, -100, -100, -100, 4, 3]])
        ce, kl, _ = supervised_and_distilled(student, s_labels, answer_logits(teacher, t_labels))
        (ce+kl).backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0)
        bad = [(torch.randn(2, 12), torch.tensor([5, 3]))]
        with self.assertRaisesRegex(ValueError, 'tokens differ'):
            supervised_and_distilled(student, s_labels, bad)
        with self.assertRaisesRegex(ValueError, 'Empty shifted'):
            answer_logits(student, torch.full((1, 5), -100))

    def test_unreliable_text_teacher_is_gated_out(self):
        labels = torch.tensor([[-100, 4, 3]])
        student = torch.zeros(1, 3, 8, requires_grad=True)
        good = student.detach().clone()
        good[0, 0, 4] = good[0, 1, 3] = 10.
        teacher = [(torch.zeros(2, 8), torch.tensor([4, 3]))]
        _, kl, gate = supervised_and_distilled(good.requires_grad_(), labels, teacher)
        self.assertEqual(float(gate), 0)
        self.assertEqual(float(kl.detach()), 0)

    @staticmethod
    def raw_pair(scene='a', task='care', split='train', caption='A real scene.'):
        return [{'id': f'{scene}::{task}::{m}', 'scene_id': scene, 'task': task, 'split': split,
                 'modality': m, 'caption': caption if m == 'text' else None,
                 'caption_index': 0 if m == 'text' else None,
                 'target_json': 'answer that must never condition the image'} for m in ('image', 'text')]

    def test_caption_index_uses_scene_information_and_ignores_gold_question(self):
        raw = self.raw_pair()+self.raw_pair(scene='b', caption='A different scene.')
        actual = caption_index(raw, split='train')
        self.assertNotEqual(actual['a'], actual['b'])
        for r in raw:
            r['target_json'] = 'changed gold'
            r['question'] = 'changed question'
        self.assertEqual(actual, caption_index(raw, split='train'))
        with self.assertRaisesRegex(ValueError, 'Cross-split'):
            caption_index(raw+self.raw_pair(scene='heldout', split='test'), split='train')
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            caption_index(raw[:1], split='train')
        with self.assertRaisesRegex(ValueError, 'captions differ'):
            caption_index(self.raw_pair()+self.raw_pair(task='fairness', caption='wrong'), split='train')

    def test_teacher_pairing_rejects_wrong_split_and_wrong_answer(self):
        pair = self.raw_pair()
        self.assertEqual(paired_text_rows(pair[:1], pair), pair[1:])
        for key, value in [('split', 'test'), ('target_json', 'other')]:
            broken = copy.deepcopy(pair)
            broken[1][key] = value
            with self.assertRaises(ValueError):
                paired_text_rows(broken[:1], broken)

    def test_serialization_and_generate_both_inference_modes(self):
        model, block = self.runtime()
        with torch.no_grad():
            block.adapter.residual.up.weight.normal_()
        inputs = self.batch()
        inputs.pop('labels')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scene'
            save_scene_guidance(path, block, {'baseline_adapter_sha256': 'test'})
            state = copy.deepcopy(block.adapter.state_dict())
            with torch.no_grad():
                block.adapter.residual.up.weight.zero_()
            load_scene_guidance(path, block)
            for key, value in block.adapter.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            for context in (SceneContext(), SceneContext('paired', torch.randn(1, 16))):
                with block.conditioned(context):
                    generated = model.generate(**inputs, max_new_tokens=2, pad_token_id=1, eos_token_id=3)
                    self.assertEqual(generated.shape[0], 2)
            with (path/'guidance.safetensors').open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_scene_guidance(path, block)

    def test_independent_inference_rejects_caption_access_before_model_runs(self):
        args = Namespace(eval_batch_size=1)
        with self.assertRaisesRegex(ValueError, 'must not receive'):
            runner.predictions(args, None, None, None, None, [], enabled=True,
                split='test', smoke=False, captions={'scene': 'leaked caption'})
        with self.assertRaisesRegex(ValueError, 'Mixed or missing'):
            runner.score([], [{'evaluation_protocol': INDEPENDENT}, {'evaluation_protocol': PAIRED}])
        with self.assertRaisesRegex(ValueError, 'Caption access'):
            runner.score([], [{'evaluation_protocol': INDEPENDENT, 'external_image_caption_access': True}])

    def test_selection_optimizes_ticr_with_quality_guards(self):
        def metric(ticr=.9, image=.8, ba=.8, joint=.7, invalid=0):
            return {'overall': {'primary_macro': {'ticr': ticr, 'image_accuracy': image,
                'image_balanced_accuracy': ba, 'joint_accuracy': joint},
                'per_task': {'care': {'image': {'invalid_outputs': invalid}}}}}
        b = metric()
        self.assertGreater(runner.scene_selection_key(metric(ticr=.91), b), runner.scene_selection_key(b, b))
        for kwargs in ({'image': .79}, {'ba': .79}, {'joint': .69}, {'invalid': 1}):
            self.assertIsNone(runner.scene_selection_key(metric(ticr=1., **kwargs), b))

    def test_invalid_scene_configuration_and_context_fail_closed(self):
        for kwargs in ({'rank': 0}, {'residual_scale': float('nan')}, {'condition_source': 'gold_answer'}):
            with self.assertRaises(ValueError):
                SceneConfig(**kwargs)
        with self.assertRaises(ValueError):
            SceneContext('paired')
        with self.assertRaises(ValueError):
            SceneContext('unknown')
        model, block = self.runtime()
        with self.assertRaises(ValueError):
            with block.conditioned(SceneContext('paired', torch.zeros(1, 16))):
                model(**self.batch())
        self.assertFalse(block._active)


if __name__ == '__main__':
    unittest.main()
