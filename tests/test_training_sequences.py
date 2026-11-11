"""Regression checks for assistant-only loss without loading model weights."""

import unittest

from moral_llava.train import (
    build_prompt,
    build_supervised_sequence,
    pad_supervised_batch,
    unpad_input_ids,
)


class TrainingSequenceTests(unittest.TestCase):
    def sequence(self, prompt=None, answer=None, **options):
        return build_supervised_sequence(
            [1, 7, 7, 8] if prompt is None else prompt,
            [10, 11] if answer is None else answer,
            eos_token_id=2,
            image_token_id=7,
            expected_image_tokens=2,
            max_length=options.pop("max_length", 32),
            **options,
        )

    def test_prompt_and_image_tokens_are_not_supervised(self):
        sample = self.sequence()
        self.assertEqual(sample["input_ids"], [1, 7, 7, 8, 10, 11, 2])
        self.assertEqual(sample["labels"], [-100, -100, -100, -100, 10, 11, 2])

    def test_eos_is_supervised_even_if_it_is_padding_token(self):
        short = self.sequence(answer=[10])
        long = self.sequence(answer=[10, 11, 12])
        batch = pad_supervised_batch([short, long], pad_token_id=2)
        eos_position = len(short["input_ids"]) - 1
        self.assertEqual(batch["labels"][0][eos_position], 2)
        self.assertEqual(batch["labels"][0][eos_position + 1:], [-100, -100])
        self.assertEqual(batch["attention_mask"][0][eos_position + 1:], [0, 0])

    def test_processor_padding_is_removed_by_positions(self):
        self.assertEqual(unpad_input_ids([2, 2, 1, 7, 2], [0, 0, 1, 1, 1]), [1, 7, 2])

    def test_image_token_expansion_cannot_silently_be_missing(self):
        with self.assertRaises(ValueError):
            self.sequence(prompt=[1, 7, 8])

    def test_truncation_cannot_drop_image_or_answer_tokens(self):
        with self.assertRaises(ValueError):
            self.sequence(max_length=5)

    def test_image_tokens_are_not_allowed_in_the_answer(self):
        with self.assertRaises(ValueError):
            self.sequence(answer=[10, 7])

    def test_prompt_requires_exactly_one_image(self):
        prompt = build_prompt("<image>\nJudge this image.")
        self.assertTrue(prompt.startswith("USER:"))
        self.assertTrue(prompt.endswith("ASSISTANT:"))
        for invalid in ("No image.", "<image> and <image>"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_prompt(invalid)


if __name__ == "__main__":
    unittest.main()
