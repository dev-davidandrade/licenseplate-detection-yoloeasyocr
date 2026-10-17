from __future__ import annotations

import csv
from pathlib import Path

from backend.services.plate_utils import clean_text, is_valid_plate


MANIFEST_FIELDS = [
    "sample_id",
    "split",
    "source_type",
    "source_path",
    "source_group",
    "box_index",
    "crop_path",
    "suggested_label",
    "reviewed_label",
    "status",
    "notes",
]
REVIEW_STATUSES = {"pending", "reviewed", "illegible", "ignored"}


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def save_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in MANIFEST_FIELDS} for row in rows
        )
    temporary_path.replace(path)


def update_review(
    rows: list[dict[str, str]],
    sample_id: str,
    status: str,
    label: str = "",
    notes: str = "",
) -> dict[str, str]:
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Status de revisão inválido: {status}")

    cleaned_label = clean_text(label)
    if status == "reviewed" and not is_valid_plate(cleaned_label):
        raise ValueError(
            "O rótulo revisado deve seguir o formato ABC1234 ou ABC1D23."
        )
    if status != "reviewed":
        cleaned_label = ""

    for row in rows:
        if row.get("sample_id") == sample_id:
            row["reviewed_label"] = cleaned_label
            row["status"] = status
            row["notes"] = notes.strip()
            return row
    raise KeyError(f"Amostra não encontrada: {sample_id}")


def review_progress(rows: list[dict[str, str]]) -> tuple[int, int]:
    completed = sum(row.get("status") != "pending" for row in rows)
    return completed, len(rows)
