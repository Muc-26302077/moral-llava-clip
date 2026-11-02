import json
import unittest

from moral_llava.schema import (
    ALL_LABELS,
    FOUNDATION_KEYS,
    PredictionParseError,
    TargetValidationError,
    labels_to_target,
    parse_prediction,
    render_answer,
    target_to_labels,
    validate_target,
)


class SchemaTests(unittest.TestCase):
    def test_every_original_label_round_trips(self):
        for label in ALL_LABELS:
            with self.subTest(label=label):
                self.assertEqual(target_to_labels(labels_to_target([label])), {label})

    def test_missing_pole_is_not_annotated(self):
        target = labels_to_target(["Care", "Betrayal"])
        self.assertEqual(target["care"], "virtue")
        self.assertEqual(target["loyalty"], "vice")
        self.assertEqual(target["fairness"], "not_annotated")
        self.assertFalse(target["non_moral"])

    def test_non_moral_preserves_absence(self):
        target = labels_to_target(["non-moral"])
        self.assertTrue(target["non_moral"])
        self.assertTrue(all(target[key] == "not_annotated" for key in FOUNDATION_KEYS))
        self.assertIn("未标注不等于已确认中性", render_answer(target))

    def test_bad_source_labels_are_rejected(self):
        for labels in ([], ["care"], ["Care", "Harm"], ["Care", "Care"], ["non-moral", "Care"]):
            with self.subTest(labels=labels), self.assertRaises(TargetValidationError):
                labels_to_target(labels)

    def test_valid_json_and_single_fence(self):
        target = labels_to_target(["Care", "Respect"])
        encoded = json.dumps(target)
        self.assertEqual(parse_prediction(encoded), target)
        self.assertEqual(parse_prediction("```json\n" + encoded + "\n```"), target)

    def test_prose_or_partial_json_is_not_repaired(self):
        text = json.dumps(labels_to_target(["Harm"]))
        for invalid in ("Here is my answer: " + text, text + " more text", text[:-1]):
            with self.subTest(invalid=invalid), self.assertRaises(PredictionParseError):
                parse_prediction(invalid)

    def test_duplicate_json_keys_are_rejected(self):
        text = json.dumps(labels_to_target(["Care"]))
        text = text.replace('{', '{"care":"vice",', 1)
        with self.assertRaises(TargetValidationError):
            parse_prediction(text)

    def test_wrong_schema_cannot_silently_be_coerced(self):
        base = labels_to_target(["Care"])
        variants = [
            {**base, "care": "neutral"},
            {**base, "non_moral": "false"},
            {**base, "non_moral": True},
            {**base, "explanation": "invented"},
            {key: value for key, value in base.items() if key != "authority"},
        ]
        for target in variants:
            with self.subTest(target=target), self.assertRaises(TargetValidationError):
                validate_target(target)


if __name__ == "__main__":
    unittest.main()
