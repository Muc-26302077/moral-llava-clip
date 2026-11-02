"""只用训练集选择每维众数，再用统一评估器计算测试集常量基线。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moral_llava.infer import format_prediction
from moral_llava.schema import FOUNDATION_KEYS, STATES, parse_prediction, validate_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.json"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/majority.jsonl"))
    parser.add_argument("--metrics", type=Path, default=Path("runs/majority_metrics.json"))
    parser.add_argument("--metadata", type=Path, default=Path("runs/majority_metadata.json"))
    args = parser.parse_args()

    train_bytes = args.train.read_bytes()
    train = json.loads(train_bytes)
    if not isinstance(train, list) or not train:
        raise ValueError("训练集必须是非空 JSON 数组。")
    counts = {key: Counter({state: 0 for state in STATES}) for key in FOUNDATION_KEYS}
    for row in train:
        answers = [message["value"] for message in row["conversations"] if message["from"] == "gpt"]
        if len(answers) != 1:
            raise ValueError(f"{row['id']}: 必须恰有一个训练答案。")
        target = parse_prediction(answers[0])
        for key in FOUNDATION_KEYS:
            counts[key][target[key]] += 1

    # max 遇到平票会取 STATES 中最先出现的状态，不读取测试标签选类别。
    chosen = {key: max(STATES, key=lambda state: counts[key][state]) for key in FOUNDATION_KEYS}
    target = validate_target({
        **chosen,
        "non_moral": all(state == "not_annotated" for state in chosen.values()),
    })
    raw_prediction = json.dumps(target, separators=(",", ":"))

    test_bytes = args.data.read_bytes()
    test = json.loads(test_bytes)
    if not isinstance(test, list) or not test:
        raise ValueError("测试集必须是非空 JSON 数组。")
    # 这里只取测试 ID；真值留给现有评估器，不参与常量判断的选择。
    predictions = [format_prediction(row["id"], raw_prediction) for row in test]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    metadata = {
        "selection_source": "training labels only; per-foundation mode",
        "training_data": str(args.train),
        "training_data_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "num_training_samples": len(train),
        "state_order_and_tie_break": list(STATES),
        "training_foundation_counts": {key: dict(counts[key]) for key in FOUNDATION_KEYS},
        "chosen_foundation_states": chosen,
        "constant_target": target,
        "test_data": str(args.data),
        "test_data_sha256": hashlib.sha256(test_bytes).hexdigest(),
        "num_test_samples": len(test),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run([
        sys.executable, "-m", "moral_llava.evaluate",
        "--data", str(args.data), "--predictions", str(args.output),
        "--output", str(args.metrics),
    ], check=True)
    print("Training-only majority target:", json.dumps(target))


if __name__ == "__main__":
    main()
