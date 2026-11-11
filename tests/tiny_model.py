"""Tiny local Transformers fixtures; no network or pretrained weights."""

class TinyModels:
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
