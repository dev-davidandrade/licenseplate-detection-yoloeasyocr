import unittest

import numpy as np

from backend.services.recognition import PlateDetection
from backend.services.realtime import RealtimePlateSession


def detection(text: str) -> PlateDetection:
    return PlateDetection(
        box=(0, 0, 10, 10),
        detector_confidence=0.9,
        raw_text=text,
        normalized_text=text,
        ocr_confidence=0.8,
        quality_score=0.7,
        crop_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
    )


class StubRecognizer:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def analyze_image(self, _image):
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return output


class RealtimePlateSessionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((20, 20, 3), dtype=np.uint8)

    def test_limits_inference_frequency_without_dropping_received_count(self):
        recognizer = StubRecognizer([[detection("ABC1234")]])
        session = RealtimePlateSession(recognizer, analyses_per_second=2)

        self.assertIsNotNone(session.analyze_if_due(self.image, now_seconds=0.0))
        self.assertIsNone(session.analyze_if_due(self.image, now_seconds=0.1))
        self.assertIsNotNone(session.analyze_if_due(self.image, now_seconds=0.5))

        snapshot = session.snapshot()
        self.assertEqual(snapshot.frames_received, 3)
        self.assertEqual(snapshot.frames_processed, 2)
        self.assertEqual(snapshot.frames_discarded, 1)
        self.assertEqual(recognizer.calls, 2)

    def test_uses_temporal_evidence_to_authorize_a_registered_plate(self):
        recognizer = StubRecognizer(
            [[detection("ABC1234")], [detection("ABC1234")]]
        )
        session = RealtimePlateSession(
            recognizer,
            analyses_per_second=2,
            authorized_plates={"ABC1234"},
            min_confirmations=2,
        )

        session.analyze_if_due(self.image, now_seconds=0.0)
        session.analyze_if_due(self.image, now_seconds=0.5)

        snapshot = session.snapshot()
        self.assertTrue(snapshot.decision.authorized)
        self.assertEqual(snapshot.decision.plate, "ABC1234")
        self.assertEqual(snapshot.decision.votes, 2)

    def test_reset_clears_previous_passage(self):
        recognizer = StubRecognizer([[detection("ABC1234")]])
        session = RealtimePlateSession(recognizer)
        session.analyze_if_due(self.image, now_seconds=0.0)

        session.reset()

        snapshot = session.snapshot()
        self.assertEqual(snapshot.frames_received, 0)
        self.assertEqual(snapshot.frames_processed, 0)
        self.assertEqual(snapshot.frames_discarded, 0)
        self.assertEqual(snapshot.candidates, ())

    def test_keeps_only_latest_frame_from_camera_backlog(self):
        recognizer = StubRecognizer([[detection("ABC1234")]])
        session = RealtimePlateSession(recognizer, analyses_per_second=2)

        session.analyze_if_due(
            self.image,
            now_seconds=0.0,
            received_frames=4,
        )
        session.analyze_if_due(
            self.image,
            now_seconds=0.1,
            received_frames=3,
        )

        snapshot = session.snapshot()
        self.assertEqual(snapshot.frames_received, 7)
        self.assertEqual(snapshot.frames_processed, 1)
        self.assertEqual(snapshot.frames_discarded, 6)
        self.assertEqual(recognizer.calls, 1)


if __name__ == "__main__":
    unittest.main()
