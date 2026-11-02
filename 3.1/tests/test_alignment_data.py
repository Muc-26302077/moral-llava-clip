import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from moral_llava.alignment_data import (
    AlignmentDataset, alignment_manifest_hash, build_retrieval_pool,
    load_alignment_splits, manifest_fingerprints,
)
from moral_llava.schema import ALL_LABELS
from scripts.evaluate_alignment import alignment_retrieval_metrics
from scripts.evaluate_retrieval import score_queries


def write_source(root: Path):
    rows = []
    specs = (
        ("train", "a", "red", ["Care", "Fairness"], ["first caption", "second caption", "third caption"]),
        ("train", "b", "orange", ["Harm"], ["one caption"]),
        ("validation", "c", "green", ["non-moral"], ["validation caption"]),
        ("test", "d", "blue", ["Sanctity"], ["test one", "test two"]),
        ("test", "e", "purple", ["non-moral"], ["test three"]),
    )
    for split, identifier, color, labels, captions in specs:
        path = root / split / f"{identifier}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color).save(path)
        rows.append({
            "id": identifier, "split": split, "image_path": f"{split}/{identifier}.png",
            "moral_labels": labels, "captions": captions,
        })
    (root / "smid_manifest.json").write_text(json.dumps(rows[:3]), encoding="utf-8")
    (root / "test/smid_manifest.json").write_text(json.dumps(rows[3:]), encoding="utf-8")
    return rows


class AlignmentDataTests(unittest.TestCase):
    def test_source_split_and_eleven_labels_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_source(root)
            splits = load_alignment_splits(root)
            self.assertEqual({key: [row["id"] for row in rows] for key, rows in splits.items()}, {
                "train": ["a", "b"], "validation": ["c"], "test": ["d", "e"],
            })
            care = splits["train"][0]
            self.assertEqual(care["captions"], ("first caption", "second caption", "third caption"))
            self.assertEqual(len(care["label_vector"]), 11)
            self.assertEqual({ALL_LABELS[index] for index, value in enumerate(care["label_vector"]) if value}, {"Care", "Fairness"})
            self.assertEqual(splits["validation"][0]["label_vector"][-1], 1.0)
            self.assertTrue(Path(care["image_path"]).is_absolute())

    def test_epoch_has_one_caption_per_image_and_cycles_reproducibly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_source(root)
            records = load_alignment_splits(root)["train"]
            first = AlignmentDataset(records, seed=14)
            second = AlignmentDataset(records, seed=14)
            seen = []
            for epoch in range(3):
                first.set_epoch(epoch)
                second.set_epoch(epoch)
                samples = [first[index] for index in range(len(first))]
                self.assertEqual(samples, [second[index] for index in range(len(second))])
                self.assertEqual([sample["id"] for sample in samples], ["a", "b"])
                self.assertEqual(samples[1]["caption"], "one caption")
                seen.append(samples[0]["caption"])
            self.assertEqual(set(seen), set(records[0]["captions"]))
            with self.assertRaisesRegex(ValueError, "mix original image splits"):
                AlignmentDataset(records + load_alignment_splits(root)["test"])

    def test_eval_pool_uses_all_captions_without_ten_caption_assumption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_source(root)
            pool = build_retrieval_pool(load_alignment_splits(root)["test"])
            self.assertEqual(pool["query_ids"], ["d", "e"])
            self.assertEqual(pool["captions"], ["test one", "test two", "test three"])
            self.assertEqual(pool["caption_image_ids"], ["d", "d", "e"])
            self.assertEqual(pool["caption_count_distribution"], {1: 1, 2: 1})
            self.assertEqual(pool["caption_ids"], ["d:0", "d:1", "e:0"])
            self.assertEqual(pool["caption_labels"], [{"Sanctity"}, {"Sanctity"}, {"non-moral"}])

    def test_fingerprints_change_when_source_caption_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = write_source(root)
            before = alignment_manifest_hash(root)
            self.assertEqual(set(manifest_fingerprints(root)), {"smid_manifest.json", "test/smid_manifest.json"})
            rows[0]["captions"][0] = "edited source caption"
            (root / "smid_manifest.json").write_text(json.dumps(rows[:3]), encoding="utf-8")
            self.assertNotEqual(before, alignment_manifest_hash(root))

    def test_training_split_load_does_not_read_test_manifest_or_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_source(root)
            (root / "test/smid_manifest.json").unlink()
            (root / "test/d.png").unlink()
            (root / "test/e.png").unlink()

            splits = load_alignment_splits(root, included_splits=("train", "validation"))
            hashes = manifest_fingerprints(root, included_splits=("train", "validation"))

            self.assertEqual(set(splits), {"train", "validation"})
            self.assertEqual(set(hashes), {"smid_manifest.json"})

    def test_cross_split_same_image_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_source(root)
            (root / "test/d.png").write_bytes((root / "train/a.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "Exact image bytes cross splits"):
                load_alignment_splits(root)

    def test_cross_split_id_reuse_and_path_disagreement_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = write_source(root)
            rows[3]["id"] = "a"
            (root / "test/smid_manifest.json").write_text(json.dumps(rows[3:]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate image id"):
                load_alignment_splits(root)
            rows[3]["id"] = "d"
            rows[3]["image_path"] = "train/a.png"
            (root / "test/smid_manifest.json").write_text(json.dumps(rows[3:]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "path and original split disagree"):
                load_alignment_splits(root)

    def test_empty_captions_and_invalid_moral_labels_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = write_source(root)
            rows[0]["captions"] = [" "]
            (root / "smid_manifest.json").write_text(json.dumps(rows[:3]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "captions must"):
                load_alignment_splits(root)
            rows[0]["captions"] = ["valid"]
            rows[0]["moral_labels"] = ["Care", "non-moral"]
            (root / "smid_manifest.json").write_text(json.dumps(rows[:3]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-moral cannot coexist"):
                load_alignment_splits(root)


class RetrievalProtocolTests(unittest.TestCase):
    @staticmethod
    def score_row(values):
        class ScoreRow(list):
            def argsort(self, *, descending):
                class Order(list):
                    def tolist(self):
                        return list(self)
                return Order(sorted(range(len(self)), key=self.__getitem__, reverse=descending))
        return ScoreRow(values)

    def test_label_relevance_and_image_pairing_have_distinct_correct_denominators(self):
        # A shares a label with C, but C is not a paired caption for A.
        scores = [self.score_row([3, 1, 4, 2]), self.score_row([4, 2, 3, 1])]
        result = score_queries(
            scores, [{"Care"}, {"Harm"}],
            [{"Care"}, {"Care"}, {"Care"}, {"Harm"}],
            ["a", "a", "c", "b"], ["a", "b"],
        )
        self.assertEqual(result["num_queries"], 2)
        self.assertEqual(result["num_candidates"], 4)
        self.assertEqual(result["label_relevance"]["r_at_1"], 0.5)
        self.assertEqual(result["same_image_pairing"]["r_at_1"], 0.0)
        self.assertEqual(result["same_image_pairing"]["r_at_5"], 1.0)
        self.assertAlmostEqual(result["label_relevance"]["map_i2t"], 7 / 12)
        self.assertAlmostEqual(result["same_image_pairing"]["map_i2t"], 3 / 8)

    def test_cross_image_moral_metric_removes_easy_own_image_captions(self):
        scores = [
            self.score_row([4, 3, 1, 2]),
            self.score_row([1, 2, 4, 3]),
            self.score_row([1, 2, 3, 4]),
        ]
        result = alignment_retrieval_metrics(
            scores, [{"Care"}, {"Care"}, {"Harm"}],
            [{"Care"}, {"Care"}, {"Care"}, {"Harm"}],
            ["a", "a", "b", "c"], ["a", "b", "c"],
        )
        # All queries rank their own caption first, while cross-image top hits are all wrong.
        label = result["label_relevance"]
        self.assertEqual(label["hit_at_1"], 1.0)
        for k in (1, 5, 10):
            self.assertEqual(label[f"r_at_{k}"], label[f"hit_at_{k}"])
        self.assertIn("not caption-level recall", label["definition"])
        cross = result["cross_image_moral_relevance"]
        self.assertEqual(cross["hit_at_1"], 0.0)
        self.assertEqual(cross["hit_at_5"], 2 / 3)
        # A has AP=1/2, B has AP=(1/2+2/3)/2, C has no other Harm image.
        self.assertAlmostEqual(cross["map_i2t"], 13 / 36)
        self.assertEqual(cross["num_queries_with_relevant_candidates"], 2)
        self.assertEqual(cross["num_queries_without_relevant_candidates"], 1)
        self.assertEqual(cross["candidate_count_after_exclusion"], {"min": 2, "max": 3})


if __name__ == "__main__":
    unittest.main()
