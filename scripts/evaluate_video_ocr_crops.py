from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import easyocr

from backend.services.plate_utils import clean_text, normalize_plate_text
from scripts.benchmark_ocr import levenshtein
from scripts.compare_easyocr import ALPHANUMERIC_ALLOWLIST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara margens de recorte do OCR em detecções de vídeo."
    )
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--video-contains", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument(
        "--top-margins",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.15, 0.2, 0.25],
    )
    parser.add_argument("--bottom-margin", type=float, default=0.05)
    parser.add_argument("--side-margin", type=float, default=0.05)
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/easyocr"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/video-ocr-crops/results.csv"),
    )
    return parser.parse_args()


def crop_with_margins(image, top: float, bottom: float, side: float):
    height, width = image.shape[:2]
    y1 = int(height * top)
    y2 = max(y1 + 1, height - int(height * bottom))
    x1 = int(width * side)
    x2 = max(x1 + 1, width - int(width * side))
    return image[y1:y2, x1:x2]


def main() -> None:
    args = parse_args()
    expected = clean_text(args.expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))

    with args.detections.open(encoding="utf-8", newline="") as input_file:
        source_rows = [
            row
            for row in csv.DictReader(input_file)
            if args.video_contains in row["video"]
        ]
    if not source_rows:
        raise RuntimeError("Nenhuma detecção corresponde ao filtro informado.")

    video_path = args.videos / source_rows[0]["video"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir {video_path}")

    crops = []
    for row in source_rows:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ok, frame = capture.read()
        if not ok:
            continue
        x1, y1, x2, y2 = map(
            int, (row["x1"], row["y1"], row["x2"], row["y2"])
        )
        crop_bgr = frame[y1:y2, x1:x2]
        if crop_bgr.size:
            crops.append(
                (
                    row,
                    cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                )
            )
    capture.release()

    reader = easyocr.Reader(
        ["pt"],
        gpu=False,
        model_storage_directory=str(args.model_cache),
        download_enabled=False,
        verbose=False,
    )

    output_rows = []
    for top_margin in args.top_margins:
        elapsed = []
        for source_row, crop_rgb in crops:
            prepared = crop_with_margins(
                crop_rgb, top_margin, args.bottom_margin, args.side_margin
            )
            started_at = time.perf_counter()
            results = reader.recognize(
                prepared,
                allowlist=ALPHANUMERIC_ALLOWLIST,
                decoder="greedy",
                detail=1,
            )
            elapsed.append(time.perf_counter() - started_at)
            if results:
                _, raw, confidence = max(results, key=lambda item: float(item[2]))
                raw = clean_text(raw)
                confidence = float(confidence)
            else:
                raw, confidence = "", 0.0
            normalized = normalize_plate_text(raw)
            output_rows.append(
                {
                    "top_margin": top_margin,
                    "frame": source_row["frame"],
                    "time_seconds": source_row["time_seconds"],
                    "expected": expected,
                    "raw_text": raw,
                    "normalized_text": normalized,
                    "exact": raw == expected,
                    "normalized_exact": normalized == expected,
                    "edit_distance": levenshtein(expected, normalized),
                    "ocr_confidence": round(confidence, 6),
                    "ocr_seconds": round(elapsed[-1], 4),
                }
            )

        margin_rows = [row for row in output_rows if row["top_margin"] == top_margin]
        exact = sum(row["exact"] for row in margin_rows)
        normalized_exact = sum(row["normalized_exact"] for row in margin_rows)
        character_accuracy = 1 - sum(
            row["edit_distance"] for row in margin_rows
        ) / (len(expected) * len(margin_rows))
        common = Counter(
            row["normalized_text"] for row in margin_rows if row["normalized_text"]
        ).most_common(3)
        print(
            f"top={top_margin:.2f} samples={len(margin_rows)} "
            f"exact={exact / len(margin_rows):.4f} "
            f"normalized_exact={normalized_exact / len(margin_rows):.4f} "
            f"character_accuracy={character_accuracy:.4f} "
            f"mean_seconds={statistics.mean(elapsed):.4f} top_candidates={common}"
        )

    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
