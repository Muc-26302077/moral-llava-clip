"""Tiny random models check the alignment paths; no pretrained download needed."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from moral_llava.alignment import AlignmentConfig


RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft")
)


class AlignmentConfigTests(unittest.TestCase):
    def test_invalid_temperature_and_projection_are_rejected(self):
        for kwargs in ({"temperature": 0}, {"temperature": float("nan")},
                       {"projection_dim": 0}, {"temperature": 0.001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AlignmentConfig(**kwargs)


@unittest.skipUnless(RUNTIME_AVAILABLE, "Model checks require torch, transformers and peft")
class AlignmentModelTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.manual_seed(19)
        torch.set_num_threads(1)

    @staticmethod
    def tiny_llava():
        from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig
        from transformers import LlavaForConditionalGeneration

        config = LlavaConfig(
            vision_config=CLIPVisionConfig(
                hidden_size=24, intermediate_size=48, num_hidden_layers=2,
                num_attention_heads=4, image_size=8, patch_size=4,
            ),
            text_config=LlamaConfig(
                vocab_size=128, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=128, pad_token_id=1,
            ),
            image_token_index=127, image_seq_length=4,
            vision_feature_layer=-2, vision_feature_select_strategy="default",
        )
        return LlavaForConditionalGeneration(config)

    @staticmethod
    def tiny_processor():
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import CLIPImageProcessor, LlavaProcessor, PreTrainedTokenizerFast

        vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
        vocab.update({f"word{i}": i for i in range(4, 127)})
        vocab["<image>"] = 127
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
            bos_token="[BOS]", eos_token="[EOS]", additional_special_tokens=["<image>"],
        )
        return LlavaProcessor(
            image_processor=CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}),
            tokenizer=fast, patch_size=4, num_additional_image_tokens=1,
            vision_feature_select_strategy="default",
        )

    @staticmethod
    def inputs(size=8):
        import torch

        return {
            "pixel_values": torch.randn(2, 3, size, size),
            "input_ids": torch.tensor([[2, 4, 3, 1], [2, 7, 8, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        }

    def test_pooling_excludes_padding(self):
        import torch
        from moral_llava.alignment import masked_mean_pool

        hidden = torch.tensor([[[1., 3.], [3., 5.], [1000., -1000.]]])
        pooled = masked_mean_pool(hidden, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(pooled, torch.tensor([[2., 4.]]))
        with self.assertRaises(ValueError):
            masked_mean_pool(hidden, torch.zeros(1, 3, dtype=torch.long))

    def test_forward_gradients_and_freeze_policy(self):
        import torch
        from moral_llava.alignment import build_alignment_model

        model = build_alignment_model(self.tiny_llava(), AlignmentConfig(
            projection_dim=8, lora_r=2, lora_alpha=4, lora_dropout=0,
            train_projector=True,
        ))
        result = model(**self.inputs())
        self.assertEqual(result["logits_per_image"].shape, (2, 2))
        for name in ("image_embeddings", "text_embeddings"):
            torch.testing.assert_close(result[name].norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(result["logits_per_text"], result["logits_per_image"].T)
        result["logits_per_image"].square().mean().backward()
        gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
        self.assertTrue(any("lora_" in name and grad is not None and grad.abs().sum() > 0
                            for name, grad in gradients.items()))
        self.assertIsNotNone(model.image_projection.weight.grad)
        self.assertIsNotNone(model.text_projection.weight.grad)
        self.assertIsNotNone(model.logit_scale.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.llava.vision_tower.parameters()))
        self.assertTrue(any(name.startswith("llava.multi_modal_projector.")
                            for name in model.trainable_parameter_names))
        for name, parameter in model.named_parameters():
            if name.startswith("llava.language_model.") and "lora_" not in name:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)

    def test_caption_embeddings_ignore_padding_content(self):
        import torch
        from moral_llava.alignment import build_alignment_model

        model = build_alignment_model(self.tiny_llava(), AlignmentConfig(
            projection_dim=8, lora_r=2, lora_alpha=4, lora_dropout=0,
            train_logit_scale=False,
        )).eval()
        short = model.encode_texts(torch.tensor([[2, 4, 3]]), torch.ones(1, 3, dtype=torch.long))
        padded = model.encode_texts(torch.tensor([[2, 4, 3, 90, 91]]), torch.tensor([[1, 1, 1, 0, 0]]))
        torch.testing.assert_close(short, padded, rtol=1e-5, atol=1e-6)
        self.assertFalse(model.logit_scale.requires_grad)
        self.assertFalse(any(name.startswith("llava.multi_modal_projector.")
                             for name in model.trainable_parameter_names))
        with self.assertRaises(ValueError):
            model.encode_texts(torch.tensor([[2, 127, 3]]), torch.ones(1, 3, dtype=torch.long))

    def test_optional_sft_keeps_llava_image_token_path_working(self):
        import torch
        from moral_llava.alignment import build_alignment_model

        model = build_alignment_model(self.tiny_llava(), AlignmentConfig(
            projection_dim=8, lora_r=2, lora_alpha=4, lora_dropout=0,
            train_projector=True,
        ))
        ids = torch.tensor([[2, 127, 127, 127, 127, 4, 5, 3]])
        labels = ids.clone()
        labels[:, :6] = -100
        output = model.forward_sft(
            input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
            pixel_values=torch.randn(1, 3, 8, 8),
        )
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.llava.multi_modal_projector.parameters()))

    def test_checkpoint_restores_real_adapter_heads_and_projector(self):
        import torch
        from moral_llava.alignment import load_alignment_checkpoint, load_alignment_model
        from moral_llava.alignment import save_alignment_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            self.tiny_llava().save_pretrained(base)
            self.tiny_processor().save_pretrained(base)
            model, processor = load_alignment_model(
                str(base), projection_dim=8, lora_r=2, lora_alpha=4,
                lora_dropout=0, train_projector=True, local_files_only=True,
            )
            batch = self.inputs()
            optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01)
            model(**batch)["logits_per_image"].square().mean().backward()
            optimizer.step()
            model.eval()
            expected = model(**batch)
            saved = save_alignment_checkpoint(model, processor, Path(directory) / "checkpoint", {"epoch": 1})
            restored, restored_processor, metadata = load_alignment_checkpoint(saved, local_files_only=True)
            actual = restored(**batch)
            torch.testing.assert_close(actual["logits_per_image"], expected["logits_per_image"])
            self.assertEqual(restored_processor.patch_size, 4)
            self.assertEqual(metadata["metadata"]["epoch"], 1)
            self.assertEqual(restored.trainable_parameter_names, [])

    def test_clip_checkpoint_swap_updates_tower_processor_and_projector(self):
        import torch
        from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPTextConfig, CLIPVisionConfig
        from moral_llava.alignment import load_alignment_checkpoint, load_alignment_model
        from moral_llava.alignment import save_alignment_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            base, vision = Path(directory) / "base", Path(directory) / "vision"
            self.tiny_llava().save_pretrained(base)
            self.tiny_processor().save_pretrained(base)
            clip = CLIPModel(CLIPConfig.from_text_vision_configs(
                CLIPTextConfig(vocab_size=128, hidden_size=16, intermediate_size=32,
                               num_attention_heads=4, num_hidden_layers=2),
                CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                                 num_hidden_layers=2, patch_size=4, image_size=12),
                projection_dim=8,
            ))
            clip.save_pretrained(vision)
            CLIPImageProcessor(size={"shortest_edge": 12}, crop_size={"height": 12, "width": 12}).save_pretrained(vision)
            model, processor = load_alignment_model(
                str(base), vision_model_id=str(vision), projection_dim=8,
                lora_r=2, lora_alpha=4, local_files_only=True,
            )
            self.assertEqual(model.llava.config.vision_config.hidden_size, 16)
            self.assertEqual(model.llava.multi_modal_projector.linear_1.in_features, 16)
            self.assertEqual(model.llava.multi_modal_projector.linear_1.out_features, 32)
            self.assertEqual(processor.image_processor.crop_size["height"], 12)
            self.assertEqual(model.llava.config.image_seq_length, 9)
            self.assertTrue(model.alignment_config.train_projector)
            torch.testing.assert_close(
                model.llava.vision_tower.vision_model.embeddings.patch_embedding.weight,
                clip.vision_model.embeddings.patch_embedding.weight,
            )
            result = model(**self.inputs(12))
            self.assertEqual(result["image_embeddings"].shape, (2, 8))
            batch = self.inputs(12)
            model.eval()
            expected = model(**batch)["logits_per_image"]
            checkpoint = save_alignment_checkpoint(model, processor, Path(directory) / "swapped")
            restored, restored_processor, _ = load_alignment_checkpoint(checkpoint)
            self.assertEqual(restored_processor.image_processor.crop_size["height"], 12)
            torch.testing.assert_close(restored(**batch)["logits_per_image"], expected)


if __name__ == "__main__":
    unittest.main()
