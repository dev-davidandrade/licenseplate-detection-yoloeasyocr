from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2

from backend.services.ocr_dataset import load_manifest, save_manifest
from backend.services.plate_utils import clean_text, is_valid_plate


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def source_name(image_path: Path) -> str:
    return re.sub(r"_(?:jpg|jpeg|png)\.rf\.[^.]+$", "", image_path.stem)


def safe_identifier(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return compact[:100] or "sample"


def load_yolo_boxes(
    label_path: Path,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, box_width, box_height = map(float, parts[:5])
        boxes.append(
            (
                max(0, round((cx - box_width / 2) * width)),
                max(0, round((cy - box_height / 2) * height)),
                min(width, round((cx + box_width / 2) * width)),
                min(height, round((cy + box_height / 2) * height)),
            )
        )
    return boxes


def expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    horizontal = round((x2 - x1) * padding_ratio)
    vertical = round((y2 - y1) * padding_ratio)
    return (
        max(0, x1 - horizontal),
        max(0, y1 - vertical),
        min(width, x2 + horizontal),
        min(height, y2 + vertical),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai recortes YOLO para revisão manual do texto das placas."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset/yolo_v2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ocr-review"),
    )
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--padding", type=float, default=0.05)
    return parser.parse_args()


def build_review_set(
    dataset: Path,
    output_dir: Path,
    splits: list[str],
    padding_ratio: float = 0.05,
) -> list[dict[str, str]]:
    manifest_path = output_dir / "manifest.csv"
    existing = {
        row["sample_id"]: row
        for row in load_manifest(manifest_path)
        if row.get("sample_id")
    }
    images_output = output_dir / "images"
    images_output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for split in splits:
        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            boxes = load_yolo_boxes(labels_dir / f"{image_path.stem}.txt", width, height)
            group = source_name(image_path)
            suggested = clean_text(group) if is_valid_plate(group) else ""
            for box_index, box in enumerate(boxes, start=1):
                x1, y1, x2, y2 = expand_box(
                    box,
                    width,
                    height,
                    padding_ratio,
                )
                if x2 <= x1 or y2 <= y1:
                    continue
                sample_id = safe_identifier(f"{split}_{group}_{box_index}")
                crop_name = f"{sample_id}.jpg"
                crop_relative = Path("images") / crop_name
                cv2.imwrite(str(output_dir / crop_relative), image[y1:y2, x1:x2])
                previous = existing.get(sample_id, {})
                rows.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "source_type": "dataset",
                        "source_path": image_path.as_posix(),
                        "source_group": group,
                        "box_index": str(box_index),
                        "crop_path": crop_relative.as_posix(),
                        "suggested_label": suggested,
                        "reviewed_label": previous.get("reviewed_label", ""),
                        "status": previous.get("status", "pending"),
                        "notes": previous.get("notes", ""),
                    }
                )

    save_manifest(manifest_path, rows)
    return rows


def main() -> None:
    args = parse_args()
    rows = build_review_set(
        args.dataset,
        args.output_dir,
        args.splits,
        args.padding,
    )
    suggestions = sum(bool(row["suggested_label"]) for row in rows)
    print(f"amostras={len(rows)}")
    print(f"sugestoes_por_nome={suggestions}")
    print(f"manifesto={args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
