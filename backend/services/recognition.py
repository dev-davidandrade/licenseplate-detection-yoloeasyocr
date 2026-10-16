from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from backend.services.plate_utils import (
    PLATE_CHARACTER_COUNT,
    clean_text,
    has_complete_plate_length,
    is_valid_plate,
    normalize_plate_text,
    plate_character_count,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "backend" / "weights" / "weights_best.pt"
DEFAULT_OCR_CACHE = PROJECT_ROOT / ".cache" / "easyocr"
DEFAULT_ULTRALYTICS_CACHE = PROJECT_ROOT / ".cache" / "ultralytics"
ALPHANUMERIC_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
RecognitionStatus = Literal["identified", "partial", "invalid", "unreadable"]


@dataclass
class PlateDetection:
    box: tuple[int, int, int, int]
    detector_confidence: float
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    quality_score: float
    crop_rgb: np.ndarray

    @property
    def valid_brazilian_plate(self) -> bool:
        return is_valid_plate(self.normalized_text)

    @property
    def character_count(self) -> int:
        return plate_character_count(self.normalized_text or self.raw_text)

    @property
    def complete_plate_length(self) -> bool:
        return has_complete_plate_length(self.normalized_text or self.raw_text)

    @property
    def recognition_status(self) -> RecognitionStatus:
        if self.valid_brazilian_plate:
            return "identified"
        if self.character_count == 0:
            return "unreadable"
        if self.character_count < PLATE_CHARACTER_COUNT:
            return "partial"
        return "invalid"

    @property
    def evidence_score(self) -> float:
        return (
            max(self.detector_confidence, 0.01)
            * max(self.ocr_confidence, 0.01)
            * (0.5 + 0.5 * self.quality_score)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "box": list(self.box),
            "detector_confidence": round(self.detector_confidence, 6),
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "ocr_confidence": round(self.ocr_confidence, 6),
            "quality_score": round(self.quality_score, 6),
            "character_count": self.character_count,
            "complete_plate_length": self.complete_plate_length,
            "recognition_status": self.recognition_status,
            "valid_brazilian_plate": self.valid_brazilian_plate,
        }


def trim_plate(
    plate_rgb: np.ndarray,
    top_ratio: float = 0.25,
    bottom_ratio: float = 0.05,
    side_ratio: float = 0.05,
) -> np.ndarray:
    if plate_rgb.size == 0:
        return plate_rgb
    height, width = plate_rgb.shape[:2]
    top = int(height * top_ratio)
    bottom = max(top + 1, height - int(height * bottom_ratio))
    left = int(width * side_ratio)
    right = max(left + 1, width - int(width * side_ratio))
    return plate_rgb[top:bottom, left:right]


def plate_crop_quality(crop_rgb: np.ndarray) -> float:
    """Returns a 0..1 ranking score based on size, sharpness and aspect ratio."""
    if crop_rgb.size == 0:
        return 0.0
    height, width = crop_rgb.shape[:2]
    if height < 2 or width < 2:
        return 0.0

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    size_score = min(1.0, (width * height) / (160 * 50))
    sharpness_score = min(1.0, sharpness / 500.0)
    aspect_ratio = width / height
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 4.0) / 4.0)
    return round(
        0.45 * size_score + 0.45 * sharpness_score + 0.10 * aspect_score,
        6,
    )


class PlateRecognizer:
    def __init__(self, detector, reader, imgsz: int = 1024, confidence: float = 0.25):
        self.detector = detector
        self.reader = reader
        self.imgsz = imgsz
        self.confidence = confidence

    def recognize_crop(self, crop_rgb: np.ndarray) -> tuple[str, str, float]:
        prepared = trim_plate(crop_rgb)
        if prepared.size == 0:
            return "", "", 0.0
        results = self.reader.recognize(
            prepared,
            allowlist=ALPHANUMERIC_ALLOWLIST,
            decoder="greedy",
            detail=1,
        )
        if not results:
            return "", "", 0.0
        _, raw_text, confidence = max(results, key=lambda item: float(item[2]))
        raw_text = clean_text(raw_text)
        return raw_text, normalize_plate_text(raw_text), float(confidence)

    def analyze_image(self, image_rgb: np.ndarray) -> list[PlateDetection]:
        if image_rgb.size == 0:
            return []
        prediction = self.detector.predict(
            image_rgb,
            imgsz=self.imgsz,
            conf=self.confidence,
            verbose=False,
        )[0]
        height, width = image_rgb.shape[:2]
        detections = []
        for result_box in prediction.boxes:
            x1, y1, x2, y2 = map(int, result_box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_rgb[y1:y2, x1:x2].copy()
            raw_text, normalized_text, ocr_confidence = self.recognize_crop(crop)
            detections.append(
                PlateDetection(
                    box=(x1, y1, x2, y2),
                    detector_confidence=float(result_box.conf[0]),
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    ocr_confidence=ocr_confidence,
                    quality_score=plate_crop_quality(crop),
                    crop_rgb=crop,
                )
            )
        return sorted(detections, key=lambda item: item.evidence_score, reverse=True)


@lru_cache(maxsize=1)
def load_default_recognizer() -> PlateRecognizer:
    DEFAULT_OCR_CACHE.mkdir(parents=True, exist_ok=True)
    DEFAULT_ULTRALYTICS_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_ULTRALYTICS_CACHE))

    import easyocr
    from ultralytics import YOLO

    detector = YOLO(str(DEFAULT_WEIGHTS))
    reader = easyocr.Reader(
        ["pt"],
        gpu=False,
        model_storage_directory=str(DEFAULT_OCR_CACHE),
        download_enabled=True,
        verbose=False,
    )
    return PlateRecognizer(detector, reader)
