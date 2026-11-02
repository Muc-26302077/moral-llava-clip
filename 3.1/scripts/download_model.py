"""Download a pinned LLaVA-1.5 snapshot and verify its weight shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
ALLOW_PATTERNS = ["*.json", "*.model", "*.safetensors"]
EXPECTED_WEIGHTS = {
    "model-00001-of-00003.safetensors": {
        "size": 4992930200,
        "sha256": "c11dbf016ee7d35ee130c19b67e20eb04873996006f424b7bcbd453b6517ee66",
    },
    "model-00002-of-00003.safetensors": {
        "size": 4957878440,
        "sha256": "46df6c6e5fad297fe7fbca4963dcf180cb1d0528cabc884d1f603311fed01328",
    },
    "model-00003-of-00003.safetensors": {
        "size": 4176137408,
        "sha256": "4f06177c37ca13944e73b57dd6d051767b950dffc12400c700dfa2509c24c697",
    },
}


def verify_weight(snapshot: Path, filename: str, expected: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": filename,
        "expected_size": expected["size"],
        "expected_sha256": expected["sha256"],
        "actual_size": None,
        "actual_sha256": None,
        "size_ok": False,
        "sha256_ok": False,
        "ok": False,
        "error": None,
    }
    try:
        path = snapshot / filename
        result["actual_size"] = path.stat().st_size
        result["size_ok"] = result["actual_size"] == expected["size"]
        digest = hashlib.sha256()
        # 分块读，大权重不用整个塞进内存。
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result["actual_sha256"] = digest.hexdigest()
        result["sha256_ok"] = result["actual_sha256"] == expected["sha256"]
        result["ok"] = result["size_ok"] and result["sha256_ok"]
        if not result["ok"]:
            result["error"] = "Weight size or SHA256 does not match the pinned snapshot."
    except OSError as error:
        result["error"] = str(error)
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only inspect the local cache; never download missing files.",
    )
    parser.add_argument("--report", help="Write the JSON report to this file as well as stdout.")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive.")
    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "allow_patterns": ALLOW_PATTERNS,
        "verify_only": args.verify_only,
        "snapshot_path": None,
        "files": [],
        "ok": False,
        "error": None,
    }
    try:
        from huggingface_hub import snapshot_download

        # 不指定 endpoint，沿用机器上的 HF_ENDPOINT 和缓存目录设置。
        snapshot_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            allow_patterns=ALLOW_PATTERNS,
            max_workers=args.max_workers,
            local_files_only=args.verify_only,
        )
        snapshot = Path(snapshot_path).resolve()
        report["snapshot_path"] = str(snapshot)
        # 三份都检查，哪一份出问题直接看报告。
        report["files"] = [
            verify_weight(snapshot, filename, expected)
            for filename, expected in EXPECTED_WEIGHTS.items()
        ]
        report["ok"] = all(result["ok"] for result in report["files"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        try:
            destination = Path(args.report).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(serialized, encoding="utf-8")
        except OSError as error:
            report["ok"] = False
            report["error"] = f"Could not write the report: {error}"
            serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    sys.stdout.write(serialized)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
