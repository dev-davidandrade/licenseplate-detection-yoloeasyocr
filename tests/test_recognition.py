import unittest

import numpy as np

from backend.services.recognition import plate_crop_quality, trim_plate


class RecognitionUtilitiesTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
