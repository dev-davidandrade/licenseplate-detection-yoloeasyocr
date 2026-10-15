from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import cv2
import easyocr

from backend.services.plate_utils import normalize_plate_text
from scripts.benchmark_ocr import PLATE_PATTERN, clean_text, levenshtein, source_name
from scripts.compare_easyocr import ALPHANUMERIC_ALLOWLIST


def trim_plate(plate_rgb):
    height, width = plate_rgb.shape[:2]
    return plate_rgb[
        int(height * 0.25) : max(int(height * 0.25) + 1, height - int(height * 0.05)),
        int(width * 0.05) : max(int(width * 0.05) + 1, width - int(width * 0.05)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia YOLO e EasyOCR em conjunto.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset/yolo_v2"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("backend/weights/weights_best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/end-to-end/results.csv"),
    )
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/easyocr"))
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ultralytics_config = Path(".cache/ultralytics").resolve()
    ultralytics_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_config))

    from ultralytics import YOLO

    detector = YOLO(str(args.weights))
    reader = easyocr.Reader(
        ["pt"],
        gpu=False,
        model_storage_directory=str(args.model_cache),
        download_enabled=False,
        verbose=False,
    )

    rows = []
    for split in args.splits:
        for image_path in sorted((args.dataset / split / "images").glob("*")):
            expected = source_name(image_path)
            if not PLATE_PATTERN.fullmatch(expected):
                continue
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue

            detector_started_at = time.perf_counter()
            result = detector.predict(image_bgr, imgsz=1024, verbose=False)[0]
            detector_seconds = time.perf_counter() - detector_started_at
            if len(result.boxes) == 0:
                rows.append(
                    {
                        "split": split,
                        "image": image_path.name,
                        "expected": expected,
                        "detected": False,
                        "detector_confidence": 0.0,
                        "raw_text": "",
                        "normalized_text": "",
                        "raw_exact": False,
                        "normalized_exact": False,
                        "edit_distance": len(expected),
                        "ocr_confidence": 0.0,
                        "detector_seconds": round(detector_seconds, 4),
                        "ocr_seconds": 0.0,
                    }
                )
                continue

            best_box = max(result.boxes, key=lambda box: float(box.conf[0]))
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            crop = trim_plate(image_rgb[y1:y2, x1:x2])

            ocr_started_at = time.perf_counter()
            ocr_results = reader.recognize(
                crop,
                allowlist=ALPHANUMERIC_ALLOWLIST,
                decoder="greedy",
                detail=1,
            )
            ocr_seconds = time.perf_counter() - ocr_started_at
            if ocr_results:
                _, raw_text, ocr_confidence = max(
                    ocr_results,
                    key=lambda item: float(item[2]),
                )
            else:
                raw_text, ocr_confidence = "", 0.0
            raw_text = clean_text(raw_text)
            normalized_text = normalize_plate_text(raw_text)
            rows.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "expected": expected,
                    "detected": True,
                    "detector_confidence": round(float(best_box.conf[0]), 6),
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "raw_exact": raw_text == expected,
                    "normalized_exact": normalized_text == expected,
                    "edit_distance": levenshtein(expected, normalized_text),
                    "ocr_confidence": round(float(ocr_confidence), 6),
                    "detector_seconds": round(detector_seconds, 4),
                    "ocr_seconds": round(ocr_seconds, 4),
                }
            )
            print(
                f"{split}: {expected} -> {raw_text or '<vazio>'} -> "
                f"{normalized_text or '<vazio>'}"
            )

    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    detected = sum(row["detected"] for row in rows)
    raw_exact = sum(row["raw_exact"] for row in rows)
    normalized_exact = sum(row["normalized_exact"] for row in rows)
    total_characters = sum(len(row["expected"]) for row in rows)
    total_distance = sum(row["edit_distance"] for row in rows)
    print(f"samples={len(rows)}")
    print(f"detection_rate={detected / len(rows):.6f}")
    print(f"raw_exact_accuracy={raw_exact / len(rows):.6f}")
    print(f"normalized_exact_accuracy={normalized_exact / len(rows):.6f}")
    print(f"normalized_character_accuracy={1 - total_distance / total_characters:.6f}")
    print(f"mean_detector_seconds={statistics.mean(row['detector_seconds'] for row in rows):.4f}")
    print(f"mean_ocr_seconds={statistics.mean(row['ocr_seconds'] for row in rows):.4f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
