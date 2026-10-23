import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.services.ocr_dataset import (
    load_manifest,
    review_progress,
    save_manifest,
    update_review,
)
from scripts.compare_easyocr import load_reviewed_samples, summarize_rows


def sample(sample_id: str = "sample-1") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "split": "test",
        "source_type": "dataset",
        "source_path": "dataset/image.jpg",
        "source_group": "ABC1234",
        "box_index": "1",
        "crop_path": "images/sample-1.jpg",
        "suggested_label": "ABC1234",
        "reviewed_label": "",
        "status": "pending",
        "notes": "",
    }


class OcrDatasetTests(unittest.TestCase):
    def test_manifest_round_trip_preserves_review_fields(self):
        test_temp = Path("artifacts/test-temp")
        test_temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_temp) as directory:
            path = Path(directory) / "manifest.csv"
            rows = [sample()]
            update_review(rows, "sample-1", "reviewed", "abc-1234", "ok")

            save_manifest(path, rows)
            loaded = load_manifest(path)

        self.assertEqual(loaded[0]["reviewed_label"], "ABC1234")
        self.assertEqual(loaded[0]["status"], "reviewed")
        self.assertEqual(loaded[0]["notes"], "ok")

    def test_rejects_invalid_reviewed_plate(self):
        rows = [sample()]

        with self.assertRaises(ValueError):
            update_review(rows, "sample-1", "reviewed", "S17")

    def test_tracks_review_progress(self):
        rows = [sample("one"), sample("two")]
        update_review(rows, "one", "illegible")

        self.assertEqual(review_progress(rows), (1, 2))

    def test_loads_only_human_reviewed_ocr_samples(self):
        test_temp = Path("artifacts/test-temp")
        test_temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_temp) as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            cv2.imwrite(
                str(images / "sample-1.jpg"),
                np.zeros((12, 30, 3), dtype=np.uint8),
            )
            rows = [sample("sample-1"), sample("pending")]
            update_review(rows, "sample-1", "reviewed", "ABC1234")
            manifest = root / "manifest.csv"
            save_manifest(manifest, rows)

            loaded = load_reviewed_samples(manifest, ["test"])

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].expected, "ABC1234")
        self.assertEqual(loaded[0].box, (0, 0, 30, 12))

    def test_summarizes_raw_and_normalized_accuracy_separately(self):
        rows = [
            {
                "expected": "ABC1234",
                "exact": False,
                "normalized_exact": True,
                "edit_distance": 1,
                "normalized_edit_distance": 0,
                "duration_seconds": 0.05,
            }
        ]

        summary = summarize_rows("experiment", "test", rows)

        self.assertEqual(summary["exact_accuracy"], 0.0)
        self.assertEqual(summary["normalized_exact_accuracy"], 1.0)
        self.assertEqual(summary["normalized_character_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
