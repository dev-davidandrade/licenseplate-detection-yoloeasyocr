import unittest

import numpy as np

from backend.services.recognition import PlateDetection, plate_crop_quality, trim_plate


class RecognitionUtilitiesTests(unittest.TestCase):
    @staticmethod
    def detection(raw_text: str, normalized_text: str) -> PlateDetection:
        return PlateDetection(
            box=(0, 0, 10, 10),
            detector_confidence=0.9,
            raw_text=raw_text,
            normalized_text=normalized_text,
            ocr_confidence=0.8,
            quality_score=0.7,
            crop_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        )

    def test_trim_plate_uses_expected_default_margins(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        trimmed = trim_plate(image)

        self.assertEqual(trimmed.shape, (70, 180, 3))

    def test_quality_prefers_a_sharp_crop_over_a_flat_crop(self):
        flat = np.full((50, 160, 3), 127, dtype=np.uint8)
        checkerboard = np.indices((50, 160)).sum(axis=0) % 2 * 255
        sharp = np.repeat(checkerboard[:, :, None], 3, axis=2).astype(np.uint8)

        self.assertGreater(plate_crop_quality(sharp), plate_crop_quality(flat))

    def test_empty_crop_has_zero_quality(self):
        empty = np.empty((0, 0, 3), dtype=np.uint8)

        self.assertEqual(plate_crop_quality(empty), 0.0)

    def test_marks_short_ocr_result_as_partial(self):
        detection = self.detection("S17", "S17")

        self.assertEqual(detection.character_count, 3)
        self.assertFalse(detection.complete_plate_length)
        self.assertFalse(detection.valid_brazilian_plate)
        self.assertEqual(detection.recognition_status, "partial")

    def test_identifies_only_a_complete_valid_plate(self):
        detection = self.detection("ABC1D23", "ABC1D23")

        self.assertEqual(detection.character_count, 7)
        self.assertTrue(detection.complete_plate_length)
        self.assertTrue(detection.valid_brazilian_plate)
        self.assertEqual(detection.recognition_status, "identified")


if __name__ == "__main__":
    unittest.main()
