from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import time
from pathlib import Path

import easyocr

from scripts.benchmark_ocr import clean_text, levenshtein
from scripts.compare_easyocr import ALPHANUMERIC_ALLOWLIST, Sample, load_samples


TOP_RATIOS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
BOTTOM_RATIOS = [0.0, 0.05, 0.10]
SIDE_RATIOS = [0.0, 0.03, 0.05, 0.08]


def crop_with_ratios(sample: Sample, top: float, bottom: float, side: float):
    x1, y1, x2, y2 = sample.box
    crop = sample.image_rgb[y1:y2, x1:x2]
    height, width = crop.shape[:2]
    top_px = round(height * top)
    bottom_px = round(height * bottom)
    side_px = round(width * side)
    return crop[
        top_px : max(top_px + 1, height - bottom_px),
        side_px : max(side_px + 1, width - side_px),
    ]


def evaluate(reader, samples, top: float, bottom: float, side: float):
    exact = 0
    edit_distance = 0
    total_characters = 0
    durations = []
    details = []
    for sample in samples:
        crop = crop_with_ratios(sample, top, bottom, side)
        started_at = time.perf_counter()
        results = reader.recognize(
            crop,
            allowlist=ALPHANUMERIC_ALLOWLIST,
            decoder="greedy",
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
    return {
        "top_ratio": top,
        "bottom_ratio": bottom,
        "side_ratio": side,
        "samples": len(samples),
        "exact_matches": exact,
        "exact_accuracy": round(exact / len(samples), 6),
        "character_accuracy": round(1 - edit_distance / total_characters, 6),
        "mean_seconds": round(statistics.mean(durations), 4),
    }, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ajusta o recorte usado pelo EasyOCR.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset/yolo_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/easyocr-crop"))
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/easyocr"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_samples = load_samples(args.dataset, ["valid"])
    test_samples = load_samples(args.dataset, ["test"])
    reader = easyocr.Reader(
        ["pt"],
        gpu=False,
        model_storage_directory=str(args.model_cache),
        download_enabled=False,
        verbose=False,
    )

    summaries = []
    combinations = list(itertools.product(TOP_RATIOS, BOTTOM_RATIOS, SIDE_RATIOS))
    for index, (top, bottom, side) in enumerate(combinations, start=1):
        summary, _ = evaluate(reader, validation_samples, top, bottom, side)
        summaries.append(summary)
        print(
            f"[{index}/{len(combinations)}] top={top:.2f} bottom={bottom:.2f} "
            f"side={side:.2f} exact={summary['exact_accuracy']:.2%}"
        )

    ranked = sorted(
        summaries,
        key=lambda row: (
            row["exact_accuracy"],
            row["character_accuracy"],
            -row["mean_seconds"],
        ),
        reverse=True,
    )
    best = ranked[0]
    test_summary, test_details = evaluate(
        reader,
        test_samples,
        best["top_ratio"],
        best["bottom_ratio"],
        best["side_ratio"],
    )

    with (args.output_dir / "validation-ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ranked[0].keys())
        writer.writeheader()
        writer.writerows(ranked)
    with (args.output_dir / "test-details.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=test_details[0].keys())
        writer.writeheader()
        writer.writerows(test_details)

    print(f"best_validation={best}")
    print(f"held_out_test={test_summary}")


if __name__ == "__main__":
    main()
