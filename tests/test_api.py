import unittest

from fastapi import HTTPException

from backend.main import health, validate_upload_content_type


class ApiTests(unittest.TestCase):
    def test_health_endpoint(self):
        self.assertEqual(health(), {"status": "ok"})

    def test_rejects_non_image_upload(self):
        with self.assertRaises(HTTPException) as context:
            validate_upload_content_type("text/plain")

        self.assertEqual(context.exception.status_code, 415)

    def test_accepts_image_upload(self):
        validate_upload_content_type("image/jpeg")


if __name__ == "__main__":
    unittest.main()
