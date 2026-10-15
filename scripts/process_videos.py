from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import easyocr
import numpy as np

from backend.services.plate_utils import clean_text, is_valid_plate, normalize_plate_text
from scripts.compare_easyocr import ALPHANUMERIC_ALLOWLIST


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass
class Reading:
    text: str
    normalized: str
    ocr_confidence: float
    detector_confidence: float


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    last_sample: int
    readings: list[Reading] = field(default_factory=list)

    def consensus(self, normalized: bool = False) -> tuple[str, float, int]:
        scores: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for reading in self.readings:
            candidate = reading.normalized if normalized else reading.text
            if len(candidate) < 4:
                continue
            weight = max(reading.ocr_confidence, 0.05) * max(
                reading.detector_confidence, 0.05
            )
            scores[candidate] += weight
            counts[candidate] += 1
        if not scores:
            return "", 0.0, 0
        best = max(scores, key=lambda candidate: (scores[candidate], counts[candidate]))
        total = sum(scores.values())
        return best, scores[best] / total if total else 0.0, counts[best]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Processa vídeos com YOLO, EasyOCR e consenso temporal."
    )
    parser.add_argument("--input", type=Path, default=Path("videos_teste"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("backend/weights/weights_best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/video-results"),
    )
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/easyocr"))
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Executa somente a detecção para uma triagem mais rápida.",
    )
    return parser.parse_args()


def discover_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Entrada não encontrada: {input_path}")
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def trim_plate(plate_rgb: np.ndarray) -> np.ndarray:
    height, width = plate_rgb.shape[:2]
    top = int(height * 0.25)
    bottom = max(top + 1, height - int(height * 0.05))
    left = int(width * 0.05)
    right = max(left + 1, width - int(width * 0.05))
    return plate_rgb[top:bottom, left:right]


def box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def center_distance_score(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    first_center = np.array(((ax1 + ax2) / 2, (ay1 + ay2) / 2))
    second_center = np.array(((bx1 + bx2) / 2, (by1 + by2) / 2))
    distance = float(np.linalg.norm(first_center - second_center))
    scale = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1)
    return distance / scale


def associate_tracks(
    tracks: list[Track],
    boxes: list[tuple[int, int, int, int]],
    sample_index: int,
    next_track_id: int,
) -> tuple[list[Track], list[Track], int]:
    active_tracks = [track for track in tracks if sample_index - track.last_sample <= 3]
    available_track_ids = {track.track_id for track in active_tracks}
    assignments: list[Track] = []

    for box in boxes:
        candidates = []
        for track in active_tracks:
            if track.track_id not in available_track_ids:
                continue
            iou = box_iou(track.box, box)
            distance = center_distance_score(track.box, box)
            if iou >= 0.05 or distance <= 1.5:
                candidates.append((iou - 0.05 * distance, track))
        if candidates:
            track = max(candidates, key=lambda item: item[0])[1]
            available_track_ids.remove(track.track_id)
            track.box = box
            track.last_sample = sample_index
        else:
            track = Track(next_track_id, box, sample_index)
            tracks.append(track)
            active_tracks.append(track)
            next_track_id += 1
        assignments.append(track)

    return tracks, assignments, next_track_id


def read_plate(reader: easyocr.Reader, crop_rgb: np.ndarray) -> tuple[str, str, float]:
    if crop_rgb.size == 0:
        return "", "", 0.0
    trimmed = trim_plate(crop_rgb)
    if trimmed.size == 0:
        return "", "", 0.0
    results = reader.recognize(
        trimmed,
        allowlist=ALPHANUMERIC_ALLOWLIST,
        decoder="greedy",
        detail=1,
    )
    if not results:
        return "", "", 0.0
    _, raw_text, confidence = max(results, key=lambda item: float(item[2]))
    text = clean_text(raw_text)
    return text, normalize_plate_text(text), float(confidence)


def safe_stem(path: Path, index: int) -> str:
    compact = "".join(character if character.isalnum() else "_" for character in path.stem)
    compact = "_".join(part for part in compact.split("_") if part)
    return f"{index:02d}_{compact[:70]}"


def draw_label(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_y = max(22, y1 - 8)
    cv2.putText(
        frame,
        text,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def process_video(
    video_path: Path,
    video_index: int,
    detector,
    reader: easyocr.Reader | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(source_fps / args.sample_fps))
    output_fps = source_fps / stride
    stem = safe_stem(video_path, video_index)
    annotated_path = args.output_dir / f"{stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Não foi possível criar: {annotated_path}")

    rows: list[dict[str, object]] = []
    tracks: list[Track] = []
    next_track_id = 1
    frame_index = 0
    sample_index = 0
    detector_times: list[float] = []
    ocr_times: list[float] = []
    pending: list[tuple[int, float, np.ndarray]] = []

    def process_batch(batch: list[tuple[int, float, np.ndarray]]) -> None:
        nonlocal tracks, next_track_id, sample_index
        if not batch:
            return
        frames = [item[2] for item in batch]
        started_at = time.perf_counter()
        predictions = detector.predict(
            frames,
            imgsz=args.imgsz,
            conf=args.detector_confidence,
            verbose=False,
        )
        batch_seconds = time.perf_counter() - started_at
        detector_times.extend([batch_seconds / len(batch)] * len(batch))

        for (original_index, timestamp, frame), prediction in zip(
            batch, predictions, strict=True
        ):
            sample_index += 1
            boxes: list[tuple[int, int, int, int]] = []
            confidences: list[float] = []
            for result_box in prediction.boxes:
                x1, y1, x2, y2 = map(int, result_box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append((x1, y1, x2, y2))
                confidences.append(float(result_box.conf[0]))

            tracks, assignments, next_track_id = associate_tracks(
                tracks, boxes, sample_index, next_track_id
            )
            for box, detector_confidence, track in zip(
                boxes, confidences, assignments, strict=True
            ):
                x1, y1, x2, y2 = box
                raw_text = normalized_text = ""
                ocr_confidence = 0.0
                if reader is not None:
                    crop_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                    ocr_started_at = time.perf_counter()
                    raw_text, normalized_text, ocr_confidence = read_plate(
                        reader, crop_rgb
                    )
                    ocr_times.append(time.perf_counter() - ocr_started_at)
                    if raw_text:
                        track.readings.append(
                            Reading(
                                raw_text,
                                normalized_text,
                                ocr_confidence,
                                detector_confidence,
                            )
                        )

                consensus, consensus_confidence, consensus_count = track.consensus()
                normalized_consensus, _, _ = track.consensus(normalized=True)
                display_text = consensus or raw_text or "placa"
                color = (0, 200, 0) if consensus_count >= 2 else (0, 180, 255)
                draw_label(
                    frame,
                    box,
                    f"#{track.track_id} {display_text} {detector_confidence:.2f}",
                    color,
                )
                rows.append(
                    {
                        "video": video_path.name,
                        "frame": original_index,
                        "time_seconds": round(timestamp, 3),
                        "track_id": track.track_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "detector_confidence": round(detector_confidence, 6),
                        "raw_text": raw_text,
                        "normalized_text": normalized_text,
                        "valid_brazilian_plate": is_valid_plate(normalized_text),
                        "ocr_confidence": round(ocr_confidence, 6),
                        "temporal_consensus": consensus,
                        "normalized_consensus": normalized_consensus,
                        "consensus_confidence": round(consensus_confidence, 6),
                        "consensus_votes": consensus_count,
                    }
                )
            cv2.putText(
                frame,
                f"t={timestamp:.1f}s | amostra {sample_index}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            pending.append((frame_index, frame_index / source_fps, frame))
            if len(pending) >= args.batch_size:
                process_batch(pending)
                pending = []
        frame_index += 1
    process_batch(pending)
    capture.release()
    writer.release()

    track_summaries = []
    for track in tracks:
        consensus, confidence, votes = track.consensus()
        normalized_consensus, _, _ = track.consensus(normalized=True)
        track_summaries.append(
            {
                "track_id": track.track_id,
                "readings": len(track.readings),
                "consensus": consensus,
                "normalized_consensus": normalized_consensus,
                "valid_brazilian_plate": is_valid_plate(normalized_consensus),
                "confidence": round(confidence, 6),
                "votes": votes,
            }
        )

    summary: dict[str, object] = {
        "video": video_path.name,
        "resolution": f"{width}x{height}",
        "source_fps": round(source_fps, 3),
        "source_frames": source_frames,
        "duration_seconds": round(source_frames / source_fps, 3),
        "sample_fps": round(output_fps, 3),
        "sampled_frames": sample_index,
        "frames_with_detections": len({row["frame"] for row in rows}),
        "detections": len(rows),
        "tracks": len(tracks),
        "tracks_with_ocr": sum(bool(track.readings) for track in tracks),
        "mean_detector_seconds": round(statistics.mean(detector_times), 4)
        if detector_times
        else 0.0,
        "mean_ocr_seconds": round(statistics.mean(ocr_times), 4)
        if ocr_times
        else 0.0,
        "annotated_video": str(annotated_path),
        "track_summaries": track_summaries,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ultralytics_config = Path(".cache/ultralytics").resolve()
    ultralytics_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_config))

    videos = discover_videos(args.input)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    if not videos:
        raise RuntimeError(f"Nenhum vídeo compatível encontrado em {args.input}")

    from ultralytics import YOLO

    detector = YOLO(str(args.weights))
    reader = None
    if not args.skip_ocr:
        reader = easyocr.Reader(
            ["pt"],
            gpu=False,
            model_storage_directory=str(args.model_cache),
            download_enabled=False,
            verbose=False,
        )

    all_rows: list[dict[str, object]] = []
    summaries = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video.name}")
        rows, summary = process_video(video, index, detector, reader, args)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"  amostras={summary['sampled_frames']} "
            f"quadros_com_placa={summary['frames_with_detections']} "
            f"detecções={summary['detections']} tracks={summary['tracks']}"
        )

    csv_path = args.output_dir / "detections.csv"
    if all_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"CSV: {csv_path if all_rows else 'nenhuma detecção'}")
    print(f"Resumo: {summary_path}")


if __name__ == "__main__":
    main()
