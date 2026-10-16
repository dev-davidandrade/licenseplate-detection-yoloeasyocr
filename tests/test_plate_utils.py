import unittest

from backend.services.plate_utils import (
    clean_text,
    has_complete_plate_length,
    is_valid_plate,
    normalize_plate_text,
    plate_character_count,
    plate_format,
)


class PlateUtilsTests(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("abc-1d23"), "ABC1D23")

    def test_recognizes_old_and_mercosul_formats(self):
        self.assertEqual(plate_format("ABC1234"), "old")
        self.assertEqual(plate_format("ABC1D23"), "mercosul")
        self.assertIsNone(plate_format("ABC12D3"))

    def test_validates_only_supported_brazilian_formats(self):
        self.assertTrue(is_valid_plate("ABC1234"))
        self.assertTrue(is_valid_plate("ABC1D23"))
        self.assertFalse(is_valid_plate("ABC123"))

    def test_requires_exactly_seven_characters(self):
        self.assertEqual(plate_character_count("ABC-1D23"), 7)
        self.assertTrue(has_complete_plate_length("ABC-1D23"))
        self.assertFalse(has_complete_plate_length("S17"))
        self.assertFalse(has_complete_plate_length("ABC12345"))

    def test_corrects_common_old_plate_confusions(self):
        self.assertEqual(normalize_plate_text("0256477"), "OZS6477")
        self.assertEqual(normalize_plate_text("J5G9648"), "JSG9648")
        self.assertEqual(normalize_plate_text("0ZU2104"), "OZU2104")

    def test_preserves_valid_mercosul_plate(self):
        self.assertEqual(normalize_plate_text("ABC1D23"), "ABC1D23")

    def test_can_prefer_a_known_format(self):
        self.assertEqual(
            normalize_plate_text("ABC1D23", preferred_format="old"),
            "ABC1023",
        )

    def test_does_not_force_invalid_length(self):
        self.assertEqual(normalize_plate_text("ABC123"), "ABC123")


if __name__ == "__main__":
    unittest.main()
