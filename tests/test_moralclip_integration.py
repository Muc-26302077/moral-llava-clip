"""Catch silent visual reinitialization and lost dense-projector checkpoints."""
import tempfile
from pathlib import Path
import unittest

import tiny_model as fixture
from moral_llava.moralclip_integration import (
    install_vision_tower, configure_trainability, verify_frozen_vision,
    validate_adapter_metadata, sha,
)


class MoralCLIPIntegrationTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPTextConfig, CLIPVisionConfig
        self.clip = CLIPModel(CLIPConfig.from_text_vision_configs(
            CLIPTextConfig(vocab_size=128, hidden_size=16, intermediate_size=32,
                           num_attention_heads=4, num_hidden_layers=2),
            CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                             num_hidden_layers=2, patch_size=4, image_size=12), projection_dim=8))
        self.source = self.root/'moralclip'
        self.clip.save_pretrained(self.source)
        CLIPImageProcessor(size={'shortest_edge':12}, crop_size={'height':12,'width':12}).save_pretrained(self.source)
        self.checksum = sha(self.source/'model.safetensors')

    def build(self):
        from peft import LoraConfig, get_peft_model, TaskType
        from moral_llava.train import find_language_lora_targets
        import torch
        model = fixture.TinyModels.tiny_llava()
        processor = fixture.TinyModels.tiny_processor()
        integration = install_vision_tower(model, processor, self.source, expected_sha256=self.checksum)
        model.requires_grad_(False)
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
                r=2, lora_alpha=4, lora_dropout=0,
                target_modules=find_language_lora_targets(model, torch.nn.Linear),
                modules_to_save=['multi_modal_projector']))
        return model, processor, integration

    @staticmethod
    def batch():
        import torch
        # One image row followed by a pure-text row: pixels only exist for image.
        ids = torch.tensor([[2]+[127]*9+[4,5,3], [2,4,5,6,7,8,9,10,11,12,4,5,3]])
        labels = ids.clone(); labels[:,:11] = -100
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                    pixel_values=torch.randn(1,3,12,12))

    def test_exact_weights_gradients_and_projector_only_stage(self):
        import torch
        model, processor, info = self.build()
        self.assertTrue(info['all_vision_tensors_match_source'])
        self.assertEqual(info['image_tokens'],9)
        self.assertEqual(info['projector'],[16,32,32])
        self.assertEqual(processor.image_processor.crop_size['height'],12)
        names = configure_trainability(model,'projector')
        self.assertTrue(all('multi_modal_projector.modules_to_save.' in n for n in names))
        loss = model(**self.batch()).loss
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters()
                            if 'modules_to_save' in n))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if 'lora_' in n))
        self.assertTrue(verify_frozen_vision(model,info))

    def test_joint_gradients_and_dense_projector_round_trip(self):
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        model, processor, info = self.build()
        configure_trainability(model,'joint')
        batch=self.batch()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        model(**batch).loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters()
                            if 'lora_' in n))
        self.assertTrue(all(p.grad is None for p in model.get_base_model().vision_tower.parameters()))
        optimizer.step();model.eval()
        expected=model(**batch).logits
        checkpoint=self.root/'adapter'
        model.save_pretrained(checkpoint,save_embedding_layers=False)
        saved=load_file(checkpoint/'adapter_model.safetensors')
        self.assertTrue(any('multi_modal_projector.linear_1.weight' in n for n in saved))
        # Reload the identical language base; tower/projector are reconstructed.
        torch.manual_seed(42)
        # Build() used a different RNG state due to CLIP initialization. Persist
        # the underlying frozen language tensors explicitly for this tiny test.
        base=fixture.TinyModels.tiny_llava()
        base.language_model.load_state_dict({k.replace('.base_layer',''):v for k,v in
            model.get_base_model().language_model.state_dict().items() if 'lora_' not in k},strict=True)
        install_vision_tower(base,processor,self.source,expected_sha256=self.checksum)
        restored=PeftModel.from_pretrained(base,checkpoint).eval()
        torch.testing.assert_close(restored(**batch).logits,expected,atol=1e-6,rtol=1e-5)
        self.assertTrue(verify_frozen_vision(restored,info))

    def test_wrong_hash_and_incompatible_adapter_rejected(self):
        model=fixture.TinyModels.tiny_llava();processor=fixture.TinyModels.tiny_processor()
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            install_vision_tower(model,processor,self.source,expected_sha256='0'*64)
        _,_,info=self.build()
        with self.assertRaisesRegex(ValueError,'not trained'):
            validate_adapter_metadata({},info)
        with self.assertRaisesRegex(ValueError,'image_tokens'):
            validate_adapter_metadata({'moralclip_integration':dict(info,image_tokens=100)},info)


if __name__=='__main__':
    unittest.main()
