from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

from moral_llava.prepare import extract_archive, prepare_dataset
from moral_llava.schema import parse_prediction


def image_bytes(color):
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def write_fixture(path, *, duplicate_image=False):
    rows = []
    colors = ("red", "green", "blue")
    splits = ("train", "validation", "test")
    labels = (["Care"], ["non-moral"], ["Harm"])
    with ZipFile(path, "w") as archive:
        for number, (split, label, color) in enumerate(zip(splits, labels, colors)):
            row = {
                "id": f"fixture_{number}",
                "split": split,
                "source_dataset": "smid",
                "image_path": f"{split}/fixture_{number}.png",
                "moral_labels": label,
                "captions": ["CAPTION_MUST_NOT_ENTER_IMAGE_ONLY_TASK"] * 10,
            }
            rows.append(row)
            archive.writestr(
                "SMID_moralclip/" + row["image_path"],
                image_bytes("red" if duplicate_image else color),
            )
        archive.writestr("SMID_moralclip/smid_manifest.json", json.dumps(rows[:2]))
        archive.writestr("SMID_moralclip/test/smid_manifest.json", json.dumps(rows[2:]))
        archive.writestr("SMID_moralclip/missing_images.json", json.dumps([{"id": "historical_absent"}]))
        archive.writestr("SMID_moralclip/ambiguous_images.json", "[]")


class PrepareTests(unittest.TestCase):
    def test_separate_test_manifest_and_no_caption_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.zip"
            write_fixture(archive)
            report = prepare_dataset(archive, root / "raw", root / "processed")
            self.assertEqual(report["split_counts"], {"train": 1, "validation": 1, "test": 1})
            all_ids = []
            for split in ("train", "validation", "test"):
                text = (root / "processed" / f"{split}.json").read_text()
                self.assertNotIn("CAPTION_MUST_NOT_ENTER_IMAGE_ONLY_TASK", text)
                rows = json.loads(text)
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]["image"].startswith(split + "/"))
                parse_prediction(rows[0]["conversations"][1]["value"])
                all_ids.append(rows[0]["id"])
            self.assertEqual(len(set(all_ids)), 3)
            # Re-running is safe, deterministic and does not report historical missing files as current gaps.
            self.assertEqual(prepare_dataset(archive, root / "raw", root / "processed"), report)

    def test_cross_split_image_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.zip"
            write_fixture(archive, duplicate_image=True)
            with self.assertRaisesRegex(ValueError, "duplicate image bytes cross splits"):
                prepare_dataset(archive, root / "raw", root / "processed")

    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.zip"
            with ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", "must not escape")
            with self.assertRaisesRegex(ValueError, "Unsafe relative path"):
                extract_archive(archive, root / "raw")
            self.assertFalse((root / "outside.txt").exists())

    def test_existing_files_are_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.zip"
            write_fixture(archive)
            extract_archive(archive, root / "raw")
            existing = root / "raw/SMID_moralclip/train/fixture_0.png"
            existing.write_bytes(b"user modified file")
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                extract_archive(archive, root / "raw")
            self.assertEqual(existing.read_bytes(), b"user modified file")


if __name__ == "__main__":
    unittest.main()
