"""CPU end-to-end checks of real encoder replacement and generation paths."""
import copy
import tempfile
from pathlib import Path
import unittest

import torch

import test_alignment_model as fixture
from moral_llava.text_guided_vision import (
    GuidanceConfig, TextGuidedResidual, condition_token_ids, install_guidance,
    load_guidance, question_conditions, save_guidance,
)


class TextGuidedVisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    @staticmethod
    def config():
        return GuidanceConfig(vision_dim=24, text_dim=32, rank=4)

    @staticmethod
    def batch():
        ids = torch.tensor([[2, 127, 127, 127, 127, 4, 5, 3],
                            [2, 4, 5, 6, 7, 8, 5, 3]])
        labels = ids.clone()
        labels[:, :6] = -100
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                    pixel_values=torch.randn(1, 3, 8, 8))

    def test_zero_initialization_reproduces_mixed_modality_logits(self):
        model = fixture.AlignmentModelTests.tiny_llava().eval()
        batch = self.batch()
        expected = model(**batch).logits.detach()
        block, index = install_guidance(model, self.config())
        self.assertEqual(index, 0)  # penultimate output of a two-block tower
        with block.conditioned(torch.randn(1, 32)):
            actual = model(**batch).logits
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with self.assertRaisesRegex(RuntimeError, 'explicit question context'):
            model(**batch)
        trainable = [n for n,p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all('.adapter.' in n for n in trainable))

    def test_cpu_supervised_generation_loss_and_text_dependency(self):
        model = fixture.AlignmentModelTests.tiny_llava().eval()
        block, _ = install_guidance(model, self.config())
        model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.language_model.model.training=True
        batch = self.batch()
        condition = torch.randn(1, 32)
        frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW(block.adapter.parameters(), lr=.03, weight_decay=0)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            with block.conditioned(condition):
                loss = model(**batch,use_cache=False).loss + .001 * block.last_penalty
                loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in block.adapter.condition.parameters()))
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        with block.conditioned(condition):
            first = model(**batch).logits.detach()
        with block.conditioned(-condition):
            other = model(**batch).logits.detach()
        self.assertGreater(float((first[0]-other[0]).abs().max()), 1e-7)
        # Text-only row cannot change when the visual condition changes.
        torch.testing.assert_close(first[1], other[1], rtol=0, atol=0)

    def test_generate_uses_condition_and_scope_cleans_up(self):
        model = fixture.AlignmentModelTests.tiny_llava().eval()
        block, _ = install_guidance(model, self.config())
        batch = self.batch()
        batch.pop('labels')
        with block.conditioned(torch.randn(1,32)):
            output = model.generate(**batch, max_new_tokens=3, do_sample=False,
                                    pad_token_id=1, eos_token_id=3)
        self.assertEqual(output.shape[0],2)
        self.assertGreater(output.shape[1],batch['input_ids'].shape[1])
        self.assertIsNone(block._condition)
        with self.assertRaisesRegex(ValueError, 'batch/text'):
            with block.conditioned(torch.randn(2,32)):
                model(**batch)
        self.assertFalse(block._active)
        with block.conditioned(None, enabled=False):
            model(**batch)

    def test_save_reload_and_hash_validation(self):
        model = fixture.AlignmentModelTests.tiny_llava().eval()
        block, _ = install_guidance(model, self.config())
        with torch.no_grad():
            block.adapter.up.weight.normal_(std=.1)
        batch, condition = self.batch(), torch.randn(1,32)
        with block.conditioned(condition):
            expected = model(**batch).logits.detach()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'adapter'
            save_guidance(path, block, {'diagnostic_only':True})
            with torch.no_grad():
                block.adapter.up.weight.zero_()
            load_guidance(path, block)
            with block.conditioned(condition):
                torch.testing.assert_close(model(**batch).logits, expected, rtol=0, atol=0)
            with (path/'guidance.safetensors').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_guidance(path, block)

    def test_prompt_conditions_exclude_gold_completion(self):
        model = fixture.AlignmentModelTests.tiny_llava().eval()
        tokenizer = fixture.AlignmentModelTests.tiny_processor().tokenizer
        row = {'modality':'image', 'prompt':'USER: <image> word4 word5 ASSISTANT:',
               'target_json':'word6'}
        first = question_conditions(model, tokenizer, [row])
        row['target_json'] = 'word99'
        second = question_conditions(model, tokenizer, [row])
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        row['prompt'] = 'USER: <image> word90 word91 ASSISTANT:'
        self.assertFalse(torch.equal(first, question_conditions(model,tokenizer,[row])))
        for prompt in ['USER: x ASSISTANT: answer', 'USER: x ASSISTANT: answer ASSISTANT:', 'USER: <image> ASSISTANT:']:
            with self.assertRaises(ValueError):
                condition_token_ids(tokenizer,prompt)

    def test_invalid_config_and_layer_rejected(self):
        for kwargs in [{'rank':0}, {'residual_scale':float('nan')}, {'condition_source':'gold_answer'}]:
            with self.assertRaises(ValueError):
                GuidanceConfig(**kwargs)
        model = fixture.AlignmentModelTests.tiny_llava()
        with self.assertRaisesRegex(ValueError, 'selected visual'):
            install_guidance(model, GuidanceConfig(vision_dim=24,text_dim=32,rank=4,feature_layer=-1))

    def test_numpy_streaming_matches_transformers_selected_layer(self):
        # The constrained CPU feature extractor must preserve real CLIP math.
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import numpy as np
        from extract_guided_features_numpy import TensorReader, vision_features, preprocess
        from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
        from PIL import Image
        tower=CLIPVisionModel(CLIPVisionConfig(hidden_size=24,intermediate_size=48,
                  num_hidden_layers=2,num_attention_heads=4,image_size=8,patch_size=4)).eval()
        pixels=torch.randn(1,3,8,8)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            tower.save_pretrained(path)
            actual=vision_features(pixels[0].numpy(),TensorReader(path/'model.safetensors'),
                                   hidden_size=24,heads=4,patch_size=4,selected_blocks=1)
            with torch.no_grad():
                expected=tower(pixels,output_hidden_states=True).hidden_states[-2][0,1:].numpy()
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
            image=Image.fromarray(np.random.default_rng(42).integers(0,256,(17,13,3),dtype=np.uint8))
            image.save(path/'input.png')
            processor=CLIPImageProcessor(size={'shortest_edge':8},crop_size={'height':8,'width':8})
            actual_pixels=preprocess(path/'input.png',processor.to_dict())
            expected_pixels=processor(images=image,return_tensors='np')['pixel_values'][0]
            np.testing.assert_allclose(actual_pixels,expected_pixels,rtol=1e-6,atol=1e-6)

    def test_checkpoint_selection_quality_guards(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from run_text_guided_vision import selection_key
        def metric(image=.8,joint=.7,ticr=.9,invalid=0):
            return {'overall':{'primary_macro':{'image_accuracy':image,'image_balanced_accuracy':image,
                      'joint_accuracy':joint,'ticr':ticr},'per_task':{'care':{'image':{'invalid_outputs':invalid}}}}}
        baseline=metric()
        self.assertIsNone(selection_key(metric(image=.79,ticr=1),baseline))
        self.assertIsNone(selection_key(metric(invalid=1),baseline))
        self.assertGreater(selection_key(metric(joint=.71),baseline),selection_key(baseline,baseline))

    def test_original_visual_weights_verified_through_structural_wrapper(self):
        from moral_llava.text_guided_vision import verify_original_vision
        from moral_llava.moralclip_integration import sha
        model=fixture.AlignmentModelTests.tiny_llava().eval()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            model.vision_tower.save_pretrained(path)
            integration={'resolved_source':str(path),'weight_sha256':sha(path/'model.safetensors')}
            block,_=install_guidance(model,self.config())
            expected=len([k for k in model.vision_tower.state_dict() if '.adapter.' not in k])
            self.assertEqual(verify_original_vision(model,integration),expected)
            with torch.no_grad():
                block.adapter.up.weight.normal_()
            self.assertEqual(verify_original_vision(model,integration),expected)
            with torch.no_grad():
                block.base_block.layer_norm1.weight[0]+=1
            with self.assertRaisesRegex(ValueError,'Frozen visual tensor changed'):
                verify_original_vision(model,integration)


if __name__ == '__main__':
    unittest.main()
