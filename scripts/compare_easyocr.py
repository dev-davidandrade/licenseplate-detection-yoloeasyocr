from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import easyocr

from scripts.benchmark_ocr import (
    PLATE_PATTERN,
    clean_text,
    levenshtein,
    load_yolo_boxes,
    source_name,
)


ALPHANUMERIC_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@dataclass(frozen=True)
class Sample:
    split: str
    image_path: Path
    expected: str
    image_rgb: object
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Experiment:
    name: str
    crop_mode: str
    preprocess: Callable
    allowlist: str | None = ALPHANUMERIC_ALLOWLIST
    decoder: str = "greedy"


def crop_plate(sample: Sample, mode: str):
    x1, y1, x2, y2 = sample.box
    if mode == "padded_5":
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 - round(width * 0.05))
        x2 = min(sample.image_rgb.shape[1], x2 + round(width * 0.05))
        y1 = max(0, y1 - round(height * 0.05))
        y2 = min(sample.image_rgb.shape[0], y2 + round(height * 0.05))

    crop = sample.image_rgb[y1:y2, x1:x2]
    if mode == "trim_current":
        height, width = crop.shape[:2]
        crop = crop[
            int(height * 0.25) : max(int(height * 0.25) + 1, height - int(height * 0.05)),
            int(width * 0.05) : max(int(width * 0.05) + 1, width - int(width * 0.05)),
        ]
    return crop


def identity(image):
    return image


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_2x(image):
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def clahe(image):
    gray_image = gray(image)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_image)


def clahe_resize_2x(image):
    return cv2.resize(clahe(image), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def otsu_resize_2x(image):
    resized = resize_2x(gray(image))
    return cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def adaptive_resize_2x(image):
    resized = resize_2x(gray(image))
    filtered = cv2.bilateralFilter(resized, 9, 75, 75)
    return cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


EXPERIMENTS = [
    Experiment("full_raw", "full", identity, allowlist=None),
    Experiment("full_allowlist", "full", identity),
    Experiment("full_allowlist_beamsearch", "full", identity, decoder="beamsearch"),
    Experiment("full_gray_allowlist", "full", gray),
    Experiment("full_resize2_allowlist", "full", resize_2x),
    Experiment("full_clahe_allowlist", "full", clahe),
    Experiment("full_clahe_resize2_allowlist", "full", clahe_resize_2x),
    Experiment("full_otsu_resize2_allowlist", "full", otsu_resize_2x),
    Experiment("full_adaptive_resize2_allowlist", "full", adaptive_resize_2x),
    Experiment("trim_current_allowlist", "trim_current", identity),
    Experiment("padded5_allowlist", "padded_5", identity),
]


def load_samples(dataset: Path, splits: list[str]) -> list[Sample]:
    samples = []
    for split in splits:
        for image_path in sorted((dataset / split / "images").glob("*")):
            expected = source_name(image_path)
            if not PLATE_PATTERN.fullmatch(expected):
                continue
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            label_path = dataset / split / "labels" / f"{image_path.stem}.txt"
            boxes = load_yolo_boxes(label_path, image_rgb.shape[1], image_rgb.shape[0])
            if boxes:
                samples.append(Sample(split, image_path, expected, image_rgb, boxes[0]))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara configurações do EasyOCR.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset/yolo_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/easyocr"))
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/easyocr"))
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.dataset, args.splits)
    reader = easyocr.Reader(
        ["pt"],
        gpu=False,
        model_storage_directory=str(args.model_cache),
        download_enabled=False,
        verbose=False,
    )

    details = []
    summaries = []
    for experiment in EXPERIMENTS:
        durations = []
        exact = 0
        edit_distance = 0
        total_characters = 0
        for sample in samples:
            input_image = experiment.preprocess(crop_plate(sample, experiment.crop_mode))
            started_at = time.perf_counter()
            results = reader.recognize(
                input_image,
                decoder=experiment.decoder,
                allowlist=experiment.allowlist,
                detail=1,
            )
            duration = time.perf_counter() - started_at
            durations.append(duration)
            if results:
                _, raw_text, confidence = max(results, key=lambda result: float(result[2]))
            else:
                raw_text, confidence = "", 0.0
            predicted = clean_text(raw_text)
            is_exact = predicted == sample.expected
            distance = levenshtein(sample.expected, predicted)
            exact += is_exact
            edit_distance += distance
            total_characters += len(sample.expected)
            details.append(
                {
                    "experiment": experiment.name,
                    "split": sample.split,
                    "image": sample.image_path.name,
                    "expected": sample.expected,
                    "predicted": predicted,
                    "exact": is_exact,
                    "edit_distance": distance,
                    "confidence": round(float(confidence), 6),
                    "duration_seconds": round(duration, 4),
                }
            )

        summary = {
            "experiment": experiment.name,
            "samples": len(samples),
            "exact_matches": exact,
            "exact_accuracy": round(exact / len(samples), 6),
            "character_accuracy": round(1 - edit_distance / total_characters, 6),
            "mean_seconds": round(statistics.mean(durations), 4),
            "median_seconds": round(statistics.median(durations), 4),
        }
        summaries.append(summary)
        print(
            f"{experiment.name}: exact={summary['exact_accuracy']:.2%} "
            f"chars={summary['character_accuracy']:.2%} "
            f"mean={summary['mean_seconds']:.3f}s"
        )

    details_path = args.output_dir / "details.csv"
    summary_path = args.output_dir / "summary.csv"
    with details_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(
            sorted(
                summaries,
                key=lambda row: (
                    row["exact_accuracy"],
                    row["character_accuracy"],
                    -row["mean_seconds"],
                ),
                reverse=True,
            )
        )
    print(f"summary={summary_path}")
    print(f"details={details_path}")


if __name__ == "__main__":
    main()
