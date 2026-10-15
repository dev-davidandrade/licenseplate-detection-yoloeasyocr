from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import easyocr


PLATE_PATTERN = re.compile(r"^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$")


def clean_text(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def source_name(image_path: Path) -> str:
    return re.sub(r"_(?:jpg|jpeg|png)\.rf\.[^.]+$", "", image_path.stem)


def load_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, cx, cy, box_width, box_height = map(float, line.split())
        x1 = max(0, round((cx - box_width / 2) * width))
        y1 = max(0, round((cy - box_height / 2) * height))
        x2 = min(width, round((cx + box_width / 2) * width))
        y2 = min(height, round((cy + box_height / 2) * height))
        boxes.append((x1, y1, x2, y2))
    return boxes


def trim_like_current_app(plate_rgb):
    height, width = plate_rgb.shape[:2]
    cut_top = int(height * 0.25)
    cut_bottom = int(height * 0.05)
    cut_sides = int(width * 0.05)
    return plate_rgb[
        cut_top : max(cut_top + 1, height - cut_bottom),
        cut_sides : max(cut_sides + 1, width - cut_sides),
    ]


def preprocess_variants(plate_rgb):
    height, width = plate_rgb.shape[:2]
    gray = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(gray_clahe, 9, 75, 75)
    threshold = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    resized_rgb = cv2.resize(
        plate_rgb,
        (width * 2, height * 2),
        interpolation=cv2.INTER_CUBIC,
    )
    resized_gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    resized_bilateral = cv2.bilateralFilter(resized_gray, 9, 75, 75)
    resized_threshold = cv2.adaptiveThreshold(
        resized_bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return [
        gray_clahe,
        bilateral,
        threshold,
        morph,
        resized_gray,
        resized_threshold,
    ]


def select_candidate(candidates: list[tuple[str, float]]) -> tuple[str, float]:
    cleaned = [
        (clean_text(text), confidence)
        for text, confidence in candidates
        if clean_text(text) not in {"", "BR", "BRASIL", "MERCOSUL"}
    ]
    if not cleaned:
        return "", 0.0
    valid = [candidate for candidate in cleaned if 6 <= len(candidate[0]) <= 7]
    return max(valid or cleaned, key=lambda candidate: candidate[1])


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia o OCR atual em recortes anotados.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset/yolo_v2"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline/ocr.csv"))
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/easyocr"))
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_cache.mkdir(parents=True, exist_ok=True)

    reader = easyocr.Reader(
        ["pt"],
        gpu=False,
        model_storage_directory=str(args.model_cache),
        verbose=False,
    )

    rows = []
    started_at = time.perf_counter()
    for split in args.splits:
        for image_path in sorted((args.dataset / split / "images").glob("*")):
            expected = source_name(image_path)
            if not PLATE_PATTERN.fullmatch(expected):
                continue

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            label_path = args.dataset / split / "labels" / f"{image_path.stem}.txt"
            boxes = load_yolo_boxes(label_path, image_rgb.shape[1], image_rgb.shape[0])
            if not boxes:
                continue

            x1, y1, x2, y2 = boxes[0]
            crop = trim_like_current_app(image_rgb[y1:y2, x1:x2])
            candidates = []
            item_started_at = time.perf_counter()
            for variant in preprocess_variants(crop):
                for _, text, confidence in reader.readtext(variant):
                    candidates.append((text, float(confidence)))
            predicted, confidence = select_candidate(candidates)
            duration = time.perf_counter() - item_started_at
            rows.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "expected": expected,
                    "predicted": predicted,
                    "exact": predicted == expected,
                    "edit_distance": levenshtein(expected, predicted),
                    "confidence": round(confidence, 6),
                    "duration_seconds": round(duration, 4),
                }
            )
            print(f"{split}: {expected} -> {predicted or '<vazio>'}")

    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    exact = sum(row["exact"] for row in rows)
    total_characters = sum(len(row["expected"]) for row in rows)
    edit_distance = sum(row["edit_distance"] for row in rows)
    elapsed = time.perf_counter() - started_at
    print(f"samples={len(rows)}")
    print(f"exact_accuracy={exact / len(rows):.6f}")
    print(f"character_accuracy={1 - edit_distance / total_characters:.6f}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
