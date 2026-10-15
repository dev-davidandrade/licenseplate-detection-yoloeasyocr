from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.services.recognition import load_default_recognizer


def process_plate_array(image_rgb: np.ndarray) -> dict[str, object]:
    detections = load_default_recognizer().analyze_image(image_rgb)
    if not detections:
        return {
            "plate_text": "Não detectado",
            "confidence": 0.0,
            "ocr_confidence": 0.0,
            "detections": [],
        }

    best = detections[0]
    return {
        "plate_text": best.normalized_text or best.raw_text or "Não detectado",
        "raw_text": best.raw_text,
        "confidence": round(best.detector_confidence, 6),
        "ocr_confidence": round(best.ocr_confidence, 6),
        "quality_score": round(best.quality_score, 6),
        "detections": [detection.as_dict() for detection in detections],
    }


def process_plate_image(image_path: str | Path) -> dict[str, object]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return process_plate_array(image_rgb)
