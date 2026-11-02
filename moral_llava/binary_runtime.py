"""Binary QA collation using the unchanged LLaVA conditional generation model."""
from __future__ import annotations

from .train import build_supervised_sequence, pad_supervised_batch, unpad_input_ids


def inputs_for_records(processor, records):
    """Mixed batches contain pixels only for rows that have image tokens."""
    from PIL import Image, ImageOps
    images = []
    for record in records:
        if record['modality'] == 'image':
            with Image.open(record['image_path']) as im:
                images.append(ImageOps.exif_transpose(im).convert('RGB'))
    prompts = [r['prompt'] for r in records]
    if images:
        return processor(text=prompts, images=images, padding=True, truncation=False, return_tensors='pt')
    return processor.tokenizer(prompts, padding=True, truncation=False, return_tensors='pt')


class BinarySFTCollator:
    def __init__(self, processor, *, image_token_id, expected_image_tokens, max_length):
        self.processor, self.image_token_id = processor, image_token_id
        self.expected_image_tokens, self.max_length = expected_image_tokens, max_length

    def __call__(self, records):
        import torch
        self.processor.tokenizer.padding_side = 'right'
        processed = inputs_for_records(self.processor, records)
        examples = []
        for i, record in enumerate(records):
            if record['target_json'] not in ('是', '否'):
                raise ValueError('Training requires a strict binary answer')
            prompt = unpad_input_ids(processed['input_ids'][i].tolist(), processed['attention_mask'][i].tolist())
            answer = self.processor.tokenizer.encode(' '+record['target_json'], add_special_tokens=False)
            examples.append(build_supervised_sequence(prompt, answer,
                eos_token_id=self.processor.tokenizer.eos_token_id, image_token_id=self.image_token_id,
                expected_image_tokens=self.expected_image_tokens if record['modality'] == 'image' else 0,
                max_length=self.max_length))
        padded = pad_supervised_batch(examples, pad_token_id=self.processor.tokenizer.pad_token_id)
        result = {k: torch.tensor(v, dtype=torch.long) for k, v in padded.items()}
        if 'pixel_values' in processed:
            result['pixel_values'] = processed['pixel_values']
        return result
