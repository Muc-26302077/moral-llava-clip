"""Mixed image/text polarity collation; supervise only answers and EOS."""
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


class PolaritySFTCollator:
    def __init__(self, processor, *, image_token_id, expected_image_tokens, max_length):
        self.processor, self.image_token_id = processor, image_token_id
        self.expected_image_tokens, self.max_length = expected_image_tokens, max_length

    def validate_target(self, record):
        from .polarity import parse_answer
        parse_answer(record['target_json'], record['task'])

    def __call__(self, records):
        import torch
        self.processor.tokenizer.padding_side = 'right'
        processed = inputs_for_records(self.processor, records)
        examples = []
        for i, record in enumerate(records):
            self.validate_target(record)
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
