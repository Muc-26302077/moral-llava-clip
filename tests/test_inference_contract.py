"""Exercise inference boundaries without importing model dependencies."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moral_llava.infer import format_prediction
from moral_llava.train import build_prompt, load_records


class UnreadableAssistant(dict):
    """Fail if inference attempts to inspect any reference assistant field."""

    def get(self, *args, **kwargs):
        raise AssertionError("Inference must not inspect a reference assistant turn")

    def __getitem__(self, key):
        raise AssertionError("Inference must not read reference assistant values")


class InferenceContractTests(unittest.TestCase):
    def test_manifest_without_gpt_target_is_valid_for_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # load_records validates the path, but leaves image decoding to the
            # model runner; these tests need neither Pillow nor a real image.
            (root / "sample.jpg").write_bytes(b"image placeholder")
            human = "<image>\nJudge the five moral foundations."
            manifest = root / "unlabeled.json"
            manifest.write_text(json.dumps([{
                "id": "sample",
                "image": "sample.jpg",
                "conversations": [{"from": "human", "value": human}],
            }]), encoding="utf-8")
            records = load_records(manifest, root, require_targets=False)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["prompt"], build_prompt(human))
            self.assertIsNone(records[0]["target_json"])

    def test_reference_assistant_is_never_read_or_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.jpg").touch()
            manifest = root / "manifest.json"
            manifest.write_text("[]", encoding="utf-8")
            human = {"from": "human", "value": "<image>\nJudge the image."}
            record = {"id": "sample", "image": "sample.jpg", "conversations": [human]}
            guarded = {
                **record,
                "conversations": [human, UnreadableAssistant({
                    "from": "gpt", "value": "SECRET REFERENCE MUST NOT ENTER PROMPT",
                })],
            }
            with patch("moral_llava.train.json.load", side_effect=[[record], [guarded]]), patch(
                "moral_llava.schema.parse_prediction",
                side_effect=AssertionError("Inference must not parse reference labels"),
            ) as parse_reference:
                unlabeled = load_records(manifest, root, require_targets=False)
                labeled = load_records(manifest, root, require_targets=False)
            parse_reference.assert_not_called()
            self.assertEqual(labeled, unlabeled)
            self.assertIsNone(labeled[0]["target_json"])
            self.assertNotIn("SECRET REFERENCE", labeled[0]["prompt"])

    def test_bad_output_keeps_verbatim_text_without_answer(self):
        malformed = ' \n不是完整 JSON：{"care": "virtue"}\n '
        contradictory = json.dumps({
            "care": "virtue", "fairness": "not_annotated",
            "loyalty": "not_annotated", "authority": "not_annotated",
            "sanctity": "not_annotated", "non_moral": True,
        })
        for raw in (malformed, contradictory):
            with self.subTest(raw=raw):
                result = format_prediction("sample", raw)
                self.assertEqual(result["id"], "sample")
                self.assertEqual(result["raw_prediction"], raw)
                self.assertIsNone(result["prediction"])
                self.assertTrue(result["parse_error"])
                self.assertIsNone(result["answer"])

    def test_valid_prediction_produces_chinese_label_summary(self):
        target = {
            "care": "virtue", "fairness": "vice", "loyalty": "not_annotated",
            "authority": "virtue", "sanctity": "vice", "non_moral": False,
        }
        raw = json.dumps(target)
        result = format_prediction("sample", raw)
        self.assertEqual(result["raw_prediction"], raw)
        self.assertEqual(result["prediction"], target)
        self.assertIsNone(result["parse_error"])
        for judgment in (
            "关怀/伤害：关怀", "公平/欺骗：欺骗", "忠诚/背叛：未标注",
            "尊重权威/颠覆权威：尊重权威", "纯洁/污秽：污秽",
        ):
            self.assertIn(judgment, result["answer"])
        self.assertIn("未标注不等于已确认中性", result["answer"])


if __name__ == "__main__":
    unittest.main()
