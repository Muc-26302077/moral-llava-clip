"""Original SMID captions and moral labels for image/text alignment."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from moral_llava.prepare import safe_relative_path
from moral_llava.schema import ALL_LABELS, labels_to_target


SPLITS = ("train", "validation", "test")
MANIFEST_PATHS = ("smid_manifest.json", "test/smid_manifest.json")
MANIFEST_SPLITS = {
    "smid_manifest.json": ("train", "validation"),
    "test/smid_manifest.json": ("test",),
}


def _requested_splits(included_splits: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(included_splits)
    if not requested or len(set(requested)) != len(requested) or any(split not in SPLITS for split in requested):
        raise ValueError(f"included_splits must be a nonempty subset of {SPLITS}")
    return requested


def _manifest_paths(included_splits: Sequence[str]) -> tuple[str, ...]:
    requested = set(_requested_splits(included_splits))
    return tuple(name for name in MANIFEST_PATHS if requested.intersection(MANIFEST_SPLITS[name]))


def manifest_fingerprints(
    image_root: str | Path,
    *,
    included_splits: Sequence[str] = SPLITS,
) -> dict[str, str]:
    root = Path(image_root).expanduser().resolve()
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in _manifest_paths(included_splits)
    }


def alignment_manifest_hash(
    image_root: str | Path,
    *,
    included_splits: Sequence[str] = SPLITS,
) -> str:
    content = json.dumps(
        manifest_fingerprints(image_root, included_splits=included_splits),
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_alignment_splits(
    image_root: str | Path,
    *,
    included_splits: Sequence[str] = SPLITS,
    verify_images: bool = True,
    check_image_hashes: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Read selected source manifests and retain their original image-level split.

    No caption is moved between splits. Byte hashes also catch the same image
    stored under different names in the loaded splits; this is not near-duplicate QA.
    """
    root = Path(image_root).expanduser().resolve()
    requested = _requested_splits(included_splits)
    splits: dict[str, list[dict[str, Any]]] = {split: [] for split in requested}
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    image_hash_splits: dict[str, tuple[str, str]] = {}
    for name in _manifest_paths(requested):
        path = root / name
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Manifest must be an array: {path}")
        for source in raw:
            if not isinstance(source, dict):
                raise ValueError(f"Invalid manifest row in {path}")
            identifier = source.get("id")
            split = source.get("split")
            if split not in SPLITS:
                raise ValueError(f"{identifier}: invalid source split {split!r}")
            if split not in requested:
                continue
            if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
                raise ValueError(f"Missing or duplicate image id: {identifier!r}")
            seen_ids.add(identifier)
            relative = safe_relative_path(source.get("image_path"))
            if relative.parts[0] != split:
                raise ValueError(f"{identifier}: image path and original split disagree")
            image_path = (root / str(relative)).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError(f"{identifier}: image path escapes image_root")
            if image_path in seen_paths:
                raise ValueError(f"Image path is used more than once: {image_path}")
            seen_paths.add(image_path)
            if verify_images or check_image_hashes:
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing image: {image_path}")
            if verify_images:
                from PIL import Image

                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.convert("RGB").load()
            if check_image_hashes:
                with image_path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                previous = image_hash_splits.get(digest)
                if previous and previous[0] != split:
                    raise ValueError(
                        f"Exact image bytes cross splits: {previous[1]} ({previous[0]}) "
                        f"and {identifier} ({split})"
                    )
                image_hash_splits[digest] = (split, identifier)
            captions = source.get("captions")
            if (
                not isinstance(captions, list) or not captions
                or any(not isinstance(value, str) or not value.strip() for value in captions)
            ):
                raise ValueError(f"{identifier}: captions must be a nonempty list of nonempty strings")
            labels = source.get("moral_labels")
            labels_to_target(labels)
            # 保留原始11类；non-moral 也参与道德相似度计算。
            label_set = set(labels)
            ordered_labels = tuple(label for label in ALL_LABELS if label in label_set)
            splits[split].append({
                "id": identifier,
                "split": split,
                "image_path": str(image_path),
                "relative_image_path": str(relative),
                "captions": tuple(captions),
                "moral_labels": ordered_labels,
                "label_vector": tuple(float(label in label_set) for label in ALL_LABELS),
            })
    for split in requested:
        if not splits[split]:
            raise ValueError(f"Original split is empty: {split}")
        splits[split].sort(key=lambda row: row["id"])
    return splits


def sampled_caption_index(identifier: str, caption_count: int, *, epoch: int, seed: int) -> int:
    if caption_count <= 0 or epoch < 0:
        raise ValueError("caption_count must be positive and epoch must be nonnegative")
    digest = hashlib.sha256(f"{seed}\0{identifier}".encode("utf-8")).digest()
    # 同一轮每张图只取一句，换轮再往后挪；不把十条描述当十张独立图。
    return (int.from_bytes(digest[:8], "big") + epoch) % caption_count


class AlignmentDataset:
    """One item per image; rotate deterministic caption choices each epoch.

    Call set_epoch before constructing that epoch's DataLoader iterator. Keep
    persistent_workers=False so worker copies receive the new epoch value.
    """

    def __init__(self, records: Sequence[dict[str, Any]], *, seed: int = 42) -> None:
        self.records = list(records)
        self.seed = int(seed)
        self.epoch = 0
        identifiers = [row["id"] for row in self.records]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("AlignmentDataset requires nonempty, unique image ids")
        if len({row["split"] for row in self.records}) != 1:
            raise ValueError("Do not mix original image splits in one AlignmentDataset")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        caption_index = sampled_caption_index(
            row["id"], len(row["captions"]), epoch=self.epoch, seed=self.seed,
        )
        return {
            "id": row["id"],
            "split": row["split"],
            "image_path": row["image_path"],
            "caption": row["captions"][caption_index],
            "caption_index": caption_index,
            "moral_labels": row["moral_labels"],
            "label_vector": row["label_vector"],
        }


def build_retrieval_pool(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Retrieval pool must contain images")
    query_ids = [row["id"] for row in records]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Retrieval queries must have unique image ids")
    if len({row["split"] for row in records}) != 1:
        raise ValueError("Do not mix original image splits in a retrieval pool")
    captions, caption_ids, caption_labels, caption_image_ids = [], [], [], []
    for row in records:
        for index, caption in enumerate(row["captions"]):
            captions.append(caption)
            caption_ids.append(f"{row['id']}:{index}")
            caption_labels.append(set(row["moral_labels"]))
            caption_image_ids.append(row["id"])
    return {
        "query_ids": query_ids,
        "image_paths": [row["image_path"] for row in records],
        "query_labels": [set(row["moral_labels"]) for row in records],
        "captions": captions,
        "caption_ids": caption_ids,
        "caption_labels": caption_labels,
        "caption_image_ids": caption_image_ids,
        "caption_count_distribution": dict(sorted(Counter(len(row["captions"]) for row in records).items())),
    }
