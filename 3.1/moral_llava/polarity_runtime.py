"""Conditional-polarity supervision on the existing mixed-modality collator."""
from .binary_runtime import BinarySFTCollator
from .polarity import parse_answer


class PolaritySFTCollator(BinarySFTCollator):
    def validate_target(self, record):
        parse_answer(record['target_json'], record['task'])
