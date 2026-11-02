"""Semantic checks for complete-coverage moral judgment evaluation."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from moral_llava.evaluate import evaluate, main
from moral_llava.schema import ALL_LABELS, FOUNDATION_KEYS, target_to_labels


def target(**changes):
    value = {foundation: "not_annotated" for foundation in FOUNDATION_KEYS}
    value.update(changes)
    value["non_moral"] = not any(value[foundation] != "not_annotated" for foundation in FOUNDATION_KEYS)
    return value


def datum(sample_id, value):
    return {
        "id": sample_id,
        "image": f"images/{sample_id}.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nJudge the five moral foundations."},
            {"from": "gpt", "value": json.dumps(value)},
        ],
    }


def prediction(sample_id, value):
    return {
        "id": sample_id,
        "raw_prediction": json.dumps(value),
        "prediction": value,
        "parse_error": None,
        "answer": None,
    }


class EvaluateTests(unittest.TestCase):
    def test_perfect_predictions_cover_every_label_and_three_classes(self):
        targets = [target()]
        for foundation in FOUNDATION_KEYS:
            targets.extend(target(**{foundation: polarity}) for polarity in ("virtue", "vice"))
        data = [datum(str(index), value) for index, value in enumerate(targets)]
        predictions = [prediction(str(index), value) for index, value in enumerate(targets)]
        metrics = evaluate(data, reversed(predictions))
        self.assertEqual(metrics["num_samples"], 11)
        self.assertEqual(metrics["valid_output_rate"], 1.0)
        for name in ("exact_match", "samples_jaccard", "micro_f1", "macro_f1", "mean_foundation_accuracy"):
            self.assertEqual(metrics[name], 1.0, name)
        self.assertEqual(set(metrics["per_label"]), set(ALL_LABELS))
        for scores in metrics["per_label"].values():
            self.assertEqual(scores["support"], 1)
            self.assertEqual(scores["f1"], 1.0)
        for scores in metrics["per_foundation"].values():
            self.assertEqual(scores["accuracy"], 1.0)
            self.assertEqual(scores["macro_f1"], 1.0)

    def test_invalid_output_stays_in_denominator_and_adds_false_negatives(self):
        value = target(care="virtue")
        valid = prediction("valid", value)
        invalid = prediction("invalid", value)
        invalid["raw_prediction"] = "I cannot return JSON"
        metrics = evaluate([datum("valid", value), datum("invalid", value)], [valid, invalid])
        self.assertEqual(metrics["num_samples"], 2)
        self.assertEqual(metrics["valid_output_rate"], 0.5)
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["samples_jaccard"], 0.5)
        self.assertAlmostEqual(metrics["micro_f1"], 2 / 3)
        care_label = next(iter(target_to_labels(value)))
        self.assertEqual(metrics["per_label"][care_label]["support"], 2)
        self.assertEqual(metrics["per_label"][care_label]["recall"], 0.5)
        self.assertEqual(metrics["parse_failure_count"], 1)
        self.assertEqual(metrics["constraint_failure_count"], 0)
        for scores in metrics["per_foundation"].values():
            self.assertEqual(scores["accuracy"], 0.5)

    def test_valid_json_is_not_task_accuracy(self):
        truth = target(**{foundation: "virtue" for foundation in FOUNDATION_KEYS})
        wrong = target(**{foundation: "vice" for foundation in FOUNDATION_KEYS})
        metrics = evaluate([datum("one", truth)], [prediction("one", wrong)])
        self.assertEqual(metrics["valid_output_rate"], 1.0)
        for name in ("exact_match", "samples_jaccard", "micro_f1", "macro_f1", "mean_foundation_accuracy"):
            self.assertEqual(metrics[name], 0.0, name)
        for scores in metrics["per_foundation"].values():
            self.assertEqual(scores["accuracy"], 0.0)
            self.assertEqual(scores["macro_f1"], 0.0)

    def test_unknown_values_and_conflicting_non_moral_are_constraint_failures(self):
        reference = target()
        unknown = target(care="virtue")
        unknown["care"] = "unknown"
        contradiction = target(care="vice")
        contradiction["non_moral"] = True
        missing_non_moral = target()
        missing_non_moral["non_moral"] = False
        data = [datum(str(index), reference) for index in range(3)]
        predictions = [prediction(str(index), value) for index, value in enumerate((unknown, contradiction, missing_non_moral))]
        metrics = evaluate(data, predictions)
        self.assertEqual(metrics["valid_output_rate"], 0.0)
        self.assertEqual(metrics["constraint_failure_count"], 3)
        self.assertEqual(metrics["parse_failure_count"], 0)
        self.assertEqual(metrics["micro_f1"], 0.0)
        self.assertEqual(metrics["exact_match"], 0.0)
        self.assertEqual(metrics["samples_jaccard"], 0.0)
        for scores in metrics["per_foundation"].values():
            self.assertEqual(scores["accuracy"], 0.0)
            self.assertEqual(scores["macro_f1"], 0.0)
            self.assertEqual(scores["per_class"]["not_annotated"]["support"], 3)

    def test_raw_output_is_authoritative_over_cached_fields(self):
        reference = target(care="virtue")
        record = prediction("one", reference)
        record["prediction"] = target(care="vice")
        record["parse_error"] = "stale error"
        metrics = evaluate([datum("one", reference)], [record])
        self.assertEqual(metrics["exact_match"], 1.0)

    def test_duplicate_json_keys_are_constraint_failures(self):
        reference = target(care="virtue")
        duplicate_json = json.dumps(reference)[:-1] + ', "care": "vice"}'
        record = prediction("one", reference)
        record["raw_prediction"] = duplicate_json
        metrics = evaluate([datum("one", reference)], [record])
        self.assertEqual(metrics["constraint_failure_count"], 1)
        self.assertEqual(metrics["exact_match"], 0.0)
        bad_data = datum("one", reference)
        bad_data["conversations"][1]["value"] = duplicate_json
        with self.assertRaisesRegex(ValueError, "invalid reference"):
            evaluate([bad_data], [prediction("one", reference)])

    def test_missing_or_non_string_raw_output_is_parse_failure(self):
        reference = target()
        for raw in (None, {}, 1, True):
            with self.subTest(raw=raw):
                record = prediction("one", reference)
                record["raw_prediction"] = raw
                metrics = evaluate([datum("one", reference)], [record])
                self.assertEqual(metrics["parse_failure_count"], 1)
                self.assertEqual(metrics["exact_match"], 0.0)
        record = prediction("one", reference)
        del record["raw_prediction"]
        self.assertEqual(evaluate([datum("one", reference)], [record])["parse_failure_count"], 1)

    def test_prediction_coverage_must_be_exact(self):
        value = target()
        data = [datum("one", value), datum("two", value)]
        cases = (
            ([prediction("one", value)], "missing prediction ids"),
            ([prediction("one", value), prediction("two", value), prediction("extra", value)], "unexpected prediction ids"),
            ([prediction("one", value), prediction("one", value), prediction("two", value)], "Duplicate prediction id"),
        )
        for records, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                evaluate(data, records)

    def test_duplicate_data_ids_invalid_reference_and_empty_data_fail(self):
        value = target()
        with self.assertRaisesRegex(ValueError, "Duplicate data id"):
            evaluate([datum("one", value), datum("one", value)], [prediction("one", value)])
        malformed = target()
        malformed["unknown"] = "virtue"
        with self.assertRaisesRegex(ValueError, "invalid reference"):
            evaluate([datum("one", malformed)], [prediction("one", value)])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            evaluate([], [])

    def test_three_class_macro_f1_includes_zero_support_classes(self):
        reference = target()
        metrics = evaluate([datum("one", reference)], [prediction("one", reference)])
        for scores in metrics["per_foundation"].values():
            self.assertEqual(scores["accuracy"], 1.0)
            self.assertAlmostEqual(scores["macro_f1"], 1 / 3)

    def test_cli_reads_jsonl_writes_metrics_and_rejects_missing_ids(self):
        value = target()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "test.json"
            prediction_path = root / "predictions.jsonl"
            output_path = root / "metrics" / "results.json"
            data_path.write_text(json.dumps([datum("one", value)]), encoding="utf-8")
            prediction_path.write_text(json.dumps(prediction("one", value)) + "\n", encoding="utf-8")
            arguments = ["--data", str(data_path), "--predictions", str(prediction_path), "--output", str(output_path)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            metrics = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["exact_match"], 1.0)
            prediction_path.write_text("", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(arguments)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
