"""Evaluate generated moral judgments against complete LLaVA-format data.

Every expected example contributes to every aggregate metric. A generation
that cannot be parsed or violates the target schema is an incorrect example,
with an empty predicted label set, rather than an example to silently drop.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .schema import (
    ALL_LABELS,
    FOUNDATION_KEYS,
    PredictionParseError,
    TargetValidationError,
    parse_prediction,
    target_to_labels,
    validate_target,
)


FOUNDATION_CLASSES = ("virtue", "vice", "not_annotated")


def _sample_id(record: Mapping[str, Any], source: str, index: int) -> str:
    value = record.get("id")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{source} record {index}: id must be a string or integer")
    sample_id = str(value)
    if not sample_id.strip():
        raise ValueError(f"{source} record {index}: id must not be empty")
    return sample_id


def _references(data: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation data must be a nonempty LLaVA JSON array")
    references: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Data record {index}: expected a JSON object")
        sample_id = _sample_id(record, "Data", index)
        if sample_id in references:
            raise ValueError(f"Duplicate data id: {sample_id!r}")
        messages = record.get("conversations")
        if not isinstance(messages, list):
            raise ValueError(f"Data id {sample_id!r}: conversations must be an array")
        answers = [
            message.get("value")
            for message in messages
            if isinstance(message, dict) and message.get("from") == "gpt"
        ]
        if len(answers) != 1 or not isinstance(answers[0], str):
            raise ValueError(
                f"Data id {sample_id!r}: expected exactly one gpt target string"
            )
        try:
            # References are serialized targets, so enforce the same complete
            # schema as generated judgments instead of guessing absent labels.
            references[sample_id] = parse_prediction(answers[0])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Data id {sample_id!r}: invalid reference: {exc}") from exc
    return references


def _prediction_index(
    predictions: Iterable[Mapping[str, Any]], expected_ids: set[str]
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(predictions, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Prediction record {index}: expected a JSON object")
        sample_id = _sample_id(record, "Prediction", index)
        if sample_id in indexed:
            raise ValueError(f"Duplicate prediction id: {sample_id!r}")
        indexed[sample_id] = record
    # 少一条也要报出来，不然后面的分数就不是同一批样本了。
    missing = sorted(expected_ids - indexed.keys())
    unexpected = sorted(indexed.keys() - expected_ids)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing prediction ids: {missing!r}")
        if unexpected:
            details.append(f"unexpected prediction ids: {unexpected!r}")
        raise ValueError("Prediction coverage mismatch; " + "; ".join(details))
    return indexed


def _scores(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def evaluate(
    data: Sequence[Mapping[str, Any]], predictions: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score the complete dataset; raise ValueError for ambiguous ID coverage.

    Only ``raw_prediction`` determines generated labels. Cached ``prediction``,
    ``parse_error`` and ``answer`` fields are intentionally not trusted. Per
    foundation macro F1 averages the three fixed classes, assigning zero F1 to
    classes with neither true nor predicted examples. Invalid generations count
    as false negatives for their reference class in every foundation.
    """
    references = _references(data)
    indexed = _prediction_index(predictions, set(references))
    label_counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in ALL_LABELS}
    foundation_counts = {
        foundation: {
            label: {"tp": 0, "fp": 0, "fn": 0} for label in FOUNDATION_CLASSES
        }
        for foundation in FOUNDATION_KEYS
    }
    foundation_correct = dict.fromkeys(FOUNDATION_KEYS, 0)
    exact_count = 0
    jaccard_sum = 0.0
    invalid_outputs: list[dict[str, str]] = []

    for sample_id, reference in references.items():
        # 重新解析模型原话，别拿缓存的 prediction 直接算分。
        raw = indexed[sample_id].get("raw_prediction")
        prediction = None
        try:
            if not isinstance(raw, str):
                raise PredictionParseError("raw_prediction must be a string")
            prediction = validate_target(parse_prediction(raw))
        except PredictionParseError as exc:
            invalid_outputs.append({"id": sample_id, "kind": "parse", "error": str(exc)})
        except TargetValidationError as exc:
            invalid_outputs.append(
                {"id": sample_id, "kind": "constraint", "error": str(exc)}
            )

        reference_labels = set(target_to_labels(reference))
        predicted_labels = set(target_to_labels(prediction)) if prediction is not None else set()
        for label, counts in label_counts.items():
            true = label in reference_labels
            predicted = label in predicted_labels
            counts["tp"] += int(true and predicted)
            counts["fp"] += int(not true and predicted)
            counts["fn"] += int(true and not predicted)

        if prediction is not None:
            exact_count += int(prediction == reference)
            union = reference_labels | predicted_labels
            jaccard_sum += len(reference_labels & predicted_labels) / len(union) if union else 1.0
        # 格式错了也算一条失败，不能丢掉后再算分。
        for foundation, classes in foundation_counts.items():
            true_class = reference[foundation]
            predicted_class = prediction[foundation] if prediction is not None else None
            foundation_correct[foundation] += int(true_class == predicted_class)
            for label, counts in classes.items():
                true = true_class == label
                predicted = predicted_class == label
                counts["tp"] += int(true and predicted)
                counts["fp"] += int(not true and predicted)
                counts["fn"] += int(true and not predicted)

    sample_count = len(references)
    per_label = {label: _scores(**counts) for label, counts in label_counts.items()}
    totals = {key: sum(counts[key] for counts in label_counts.values()) for key in ("tp", "fp", "fn")}
    micro = _scores(**totals)
    per_foundation = {}
    for foundation, classes in foundation_counts.items():
        per_class = {label: _scores(**counts) for label, counts in classes.items()}
        per_foundation[foundation] = {
            "accuracy": foundation_correct[foundation] / sample_count,
            "macro_f1": sum(scores["f1"] for scores in per_class.values()) / len(FOUNDATION_CLASSES),
            "per_class": per_class,
        }
    parse_failures = sum(item["kind"] == "parse" for item in invalid_outputs)
    constraint_failures = sum(item["kind"] == "constraint" for item in invalid_outputs)
    return {
        "num_samples": sample_count,
        "valid_output_count": sample_count - len(invalid_outputs),
        "valid_output_rate": (sample_count - len(invalid_outputs)) / sample_count,
        "invalid_output_count": len(invalid_outputs),
        "parse_failure_count": parse_failures,
        "constraint_failure_count": constraint_failures,
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_precision": sum(scores["precision"] for scores in per_label.values()) / len(ALL_LABELS),
        "macro_recall": sum(scores["recall"] for scores in per_label.values()) / len(ALL_LABELS),
        "macro_f1": sum(scores["f1"] for scores in per_label.values()) / len(ALL_LABELS),
        "samples_jaccard": jaccard_sum / sample_count,
        "exact_match": exact_count / sample_count,
        "mean_foundation_accuracy": sum(foundation_correct.values()) / (sample_count * len(FOUNDATION_KEYS)),
        "per_label": per_label,
        "per_foundation": per_foundation,
        "invalid_outputs": invalid_outputs,
    }


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc.msg}") from exc
    return records


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="LLaVA conversations JSON array")
    parser.add_argument("--predictions", type=Path, required=True, help="Inference predictions JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Destination metrics JSON")
    args = parser.parse_args(argv)
    try:
        with args.data.open(encoding="utf-8") as handle:
            data = json.load(handle)
        metrics = evaluate(data, _load_predictions(args.predictions))
        _write_json(args.output, metrics)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(
        f"Evaluated {metrics['num_samples']} samples: "
        f"exact match={metrics['exact_match']:.4f}, "
        f"micro F1={metrics['micro_f1']:.4f}, "
        f"valid outputs={metrics['valid_output_rate']:.4f}. Metrics: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
