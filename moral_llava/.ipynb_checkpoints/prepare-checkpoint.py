"""Audit and convert the supplied SMID archive without changing its split."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import ZipFile

from .schema import ALL_LABELS, FOUNDATION_KEYS, TASK_PROMPT, labels_to_target


def safe_relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Expected a non-empty POSIX relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def extract_archive(archive, raw_dir):
    """Reject traversal/symlinks and never replace a different existing file."""
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as handle:
        members = handle.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate member paths.")
        for item in members:
            relative = safe_relative_path(item.filename)
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Archive contains a symlink: {item.filename}")
            destination = (raw_dir / str(relative)).resolve()
            if not destination.is_relative_to(raw_dir):
                raise ValueError(f"Archive path escapes extraction directory: {item.filename}")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                # 重跑时先比下内容，别把改过的原文件直接盖掉。
                with handle.open(item) as source:
                    incoming_hash = hashlib.file_digest(source, "sha256").hexdigest()
                with destination.open("rb") as existing:
                    existing_hash = hashlib.file_digest(existing, "sha256").hexdigest()
                if incoming_hash != existing_hash:
                    raise ValueError(f"Refusing to replace an existing file: {destination}")
                continue
            with handle.open(item) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
    root = raw_dir / "SMID_moralclip"
    if not (root / "smid_manifest.json").is_file():
        raise ValueError("Expected SMID_moralclip/smid_manifest.json in the archive.")
    return root


def load_records(image_root):
    image_root = Path(image_root)
    # 测试集的清单单独放在 test 里，只读根目录会漏掉这部分。
    manifests = [image_root / "smid_manifest.json", image_root / "test/smid_manifest.json"]
    records = {}
    counts = {}
    for path in manifests:
        if not path.is_file():
            raise ValueError(f"Missing manifest: {path}")
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, list):
            raise ValueError(f"Manifest must be an array: {path}")
        counts[str(path.relative_to(image_root))] = len(content)
        for row in content:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"Invalid manifest record in {path}")
            identifier = row["id"]
            if identifier in records:
                if records[identifier] != row:
                    raise ValueError(f"Conflicting or cross-split duplicate id: {identifier}")
                continue
            records[identifier] = row
    return list(records.values()), counts


def audit_records(records, image_root):
    """Check all images, original label constraints and exact cross-split duplicates."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install requirements-data.txt to audit image files.") from exc

    image_root = Path(image_root).resolve()
    digests = defaultdict(list)
    paths = set()
    split_counts = Counter()
    label_counts = {split: Counter() for split in ("train", "validation", "test")}
    states = {
        split: {key: Counter() for key in FOUNDATION_KEYS}
        for split in ("train", "validation", "test")
    }
    caption_counts = Counter()
    for row in records:
        split = row.get("split")
        if split not in label_counts:
            raise ValueError(f"Unknown split for {row['id']}: {split!r}")
        if row.get("source_dataset") != "smid":
            raise ValueError(f"Expected SMID source for {row['id']}")
        relative = safe_relative_path(row.get("image_path"))
        if relative.parts[0] != split:
            raise ValueError(f"Image path and split disagree for {row['id']}")
        path = (image_root / str(relative)).resolve()
        if not path.is_relative_to(image_root) or not path.is_file():
            raise ValueError(f"Missing or unsafe image: {row.get('image_path')}")
        if path in paths:
            raise ValueError(f"Image path is reused by multiple records: {path}")
        paths.add(path)
        target = labels_to_target(row.get("moral_labels"))
        captions = row.get("captions")
        if not isinstance(captions, list) or any(not isinstance(c, str) for c in captions):
            raise ValueError(f"Invalid captions for {row['id']}")
        caption_counts[len(captions)] += 1
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        digests[digest].append({"id": row["id"], "split": split})
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.convert("RGB").load()
        except Exception as exc:
            raise ValueError(f"Unreadable image: {path}: {exc}") from exc
        split_counts[split] += 1
        label_counts[split].update(row["moral_labels"])
        for key in FOUNDATION_KEYS:
            states[split][key][target[key]] += 1
    if any(not split_counts[split] for split in label_counts):
        raise ValueError("All of train, validation and test must contain records.")
    duplicates = [group for group in digests.values() if len(group) > 1]
    cross_split = [group for group in duplicates if len({r['split'] for r in group}) > 1]
    if cross_split:
        raise ValueError(f"Exact duplicate image bytes cross splits: {cross_split}")
    return {
        "num_images": len(records),
        "split_counts": dict(split_counts),
        "label_counts": {
            split: {label: counts[label] for label in ALL_LABELS}
            for split, counts in label_counts.items()
        },
        "foundation_state_counts": {
            split: {key: dict(counts) for key, counts in dims.items()}
            for split, dims in states.items()
        },
        "caption_count_distribution": dict(caption_counts),
        "same_split_exact_duplicates": duplicates,
        "cross_split_exact_duplicates": cross_split,
        "image_integrity_checked": True,
        "duplicate_check": "SHA-256 of file bytes; not a near-duplicate search",
    }


def make_conversation(row):
    # 只拿标签做答案，原来的 captions 不往问题里塞。
    target = labels_to_target(row["moral_labels"])
    return {
        "id": row["id"],
        "image": row["image_path"],
        "conversations": [
            {"from": "human", "value": "<image>\n" + TASK_PROMPT},
            {"from": "gpt", "value": json.dumps(target, separators=(",", ":"))},
        ],
    }


def prepare_dataset(archive, raw_dir, output_dir):
    archive = Path(archive).resolve()
    output_dir = Path(output_dir).resolve()
    image_root = extract_archive(archive, raw_dir)
    records, manifest_counts = load_records(image_root)
    report = audit_records(records, image_root)
    with archive.open("rb") as stream:
        report["archive_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    report.update(
        {
            "schema_version": 1,
            "archive": archive.name,
            "manifest_counts": manifest_counts,
            "label_semantics": "Absent poles are not_annotated, not verified neutral.",
            "training_input": "image and fixed instruction only; no source captions",
            "split_policy": "Preserve original image-level split without resampling.",
        }
    )
    for name in ("missing_images.json", "ambiguous_images.json"):
        path = image_root / name
        if path.exists():
            report["historical_" + name.removesuffix(".json")] = json.loads(
                path.read_text(encoding="utf-8")
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        conversations = [
            make_conversation(row)
            for row in sorted(records, key=lambda row: row["id"])
            if row["split"] == split
        ]
        (output_dir / f"{split}.json").write_text(
            json.dumps(conversations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("SMID_moralclip.zip"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    report = prepare_dataset(args.archive, args.raw_dir, args.output_dir)
    print(json.dumps({"num_images": report["num_images"], "split_counts": report["split_counts"]}, indent=2))
    print(f"Image root: {(args.raw_dir / 'SMID_moralclip').resolve()}")
    print(f"Conversations and audit: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
