import unittest
from unittest.mock import patch

import numpy as np

from backend.services.ocr_service import process_plate_array
from backend.services.recognition import PlateDetection


class StubRecognizer:
    def __init__(self, detections):
        self.detections = detections

    def analyze_image(self, _image):
        return self.detections


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


class OcrServiceTests(unittest.TestCase):
    def process(self, detections):
        recognizer = StubRecognizer(detections)
        with patch(
            "backend.services.ocr_service.load_default_recognizer",
            return_value=recognizer,
        ):
            return process_plate_array(np.zeros((20, 20, 3), dtype=np.uint8))

    def test_partial_text_is_not_returned_as_an_identified_plate(self):
        result = self.process([detection("S17", "S17")])

        self.assertEqual(result["plate_text"], "Não identificado")
        self.assertEqual(result["raw_text"], "S17")
        self.assertEqual(result["character_count"], 3)
        self.assertEqual(result["recognition_status"], "partial")
        self.assertFalse(result["valid_brazilian_plate"])

    def test_complete_valid_text_is_returned_as_an_identified_plate(self):
        result = self.process([detection("ABC1D23", "ABC1D23")])

        self.assertEqual(result["plate_text"], "ABC1D23")
        self.assertEqual(result["character_count"], 7)
        self.assertEqual(result["recognition_status"], "identified")
        self.assertTrue(result["valid_brazilian_plate"])

    def test_valid_plate_is_preferred_over_a_higher_ranked_partial_reading(self):
        result = self.process(
            [
                detection("S17", "S17"),
                detection("ABC1234", "ABC1234"),
            ]
        )

        self.assertEqual(result["plate_text"], "ABC1234")
        self.assertTrue(result["valid_brazilian_plate"])

    def test_no_detection_has_an_explicit_status(self):
        result = self.process([])

        self.assertEqual(result["plate_text"], "Não detectado")
        self.assertEqual(result["recognition_status"], "not_detected")
        self.assertFalse(result["valid_brazilian_plate"])


if __name__ == "__main__":
    unittest.main()
