from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from backend.services.recognition import PlateDetection, PlateRecognizer
from backend.services.temporal import (
    AuthorizationDecision,
    PlateCandidate,
    PlateReading,
    TemporalPlateAggregator,
)


@dataclass(frozen=True)
class RealtimeSnapshot:
    frames_received: int
    frames_processed: int
    frames_discarded: int
    detections: int
    evidence_accepted: int
    evidence_rejected: int
    last_processing_seconds: float
    candidates: tuple[PlateCandidate, ...]
    decision: AuthorizationDecision
    last_error: str


@dataclass(frozen=True)
class EvidenceThresholds:
    """Provisional safety gates for readings entering temporal consensus."""

    min_detector_confidence: float = 0.50
    min_ocr_confidence: float = 0.60
    min_quality_score: float = 0.50

    def __post_init__(self) -> None:
        for field_name, value in (
            ("min_detector_confidence", self.min_detector_confidence),
            ("min_ocr_confidence", self.min_ocr_confidence),
            ("min_quality_score", self.min_quality_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} deve estar entre 0 e 1")

    def accepts(self, detection: PlateDetection) -> bool:
        return (
            detection.valid_brazilian_plate
            and detection.detector_confidence >= self.min_detector_confidence
            and detection.ocr_confidence >= self.min_ocr_confidence
            and detection.quality_score >= self.min_quality_score
        )


@dataclass(frozen=True)
class RegionOfInterest:
    """Normalized fixed-camera region analyzed by the recognizer."""

    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.left < self.right <= 1.0
            and 0.0 <= self.top < self.bottom <= 1.0
        ):
            raise ValueError("A região de interesse deve estar dentro da imagem")

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = min(width - 1, max(0, round(width * self.left)))
        y1 = min(height - 1, max(0, round(height * self.top)))
        x2 = min(width, max(x1 + 1, round(width * self.right)))
        y2 = min(height, max(y1 + 1, round(height * self.bottom)))
        return x1, y1, x2, y2


class RealtimePlateSession:
    """Thread-safe state shared by Streamlit and the WebRTC callback."""

    def __init__(
        self,
        recognizer: PlateRecognizer,
        analyses_per_second: float = 2.0,
        authorized_plates: set[str] | None = None,
        min_confirmations: int = 2,
        evidence_thresholds: EvidenceThresholds | None = None,
        roi: RegionOfInterest | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.recognizer = recognizer
        self._clock = clock
        self._state_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._aggregator = TemporalPlateAggregator()
        self._started_at: float | None = None
        self._next_analysis_at = float("-inf")
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_discarded = 0
        self._detections = 0
        self._evidence_accepted = 0
        self._evidence_rejected = 0
        self._last_processing_seconds = 0.0
        self._last_error = ""
        self._evidence_thresholds = evidence_thresholds or EvidenceThresholds()
        self._roi = roi or RegionOfInterest()
        self.configure(
            analyses_per_second=analyses_per_second,
            authorized_plates=authorized_plates or set(),
            min_confirmations=min_confirmations,
            evidence_thresholds=self._evidence_thresholds,
            roi=self._roi,
        )

    def configure(
        self,
        analyses_per_second: float,
        authorized_plates: set[str],
        min_confirmations: int,
        evidence_thresholds: EvidenceThresholds | None = None,
        roi: RegionOfInterest | None = None,
    ) -> None:
        with self._state_lock:
            next_thresholds = evidence_thresholds or self._evidence_thresholds
            next_roi = roi or self._roi
            evidence_context_changed = (
                next_thresholds != self._evidence_thresholds
                or next_roi != self._roi
            )
            self._analyses_per_second = max(0.1, float(analyses_per_second))
            self._authorized_plates = set(authorized_plates)
            self._min_confirmations = max(1, int(min_confirmations))
            self._evidence_thresholds = next_thresholds
            self._roi = next_roi
            if evidence_context_changed:
                self._aggregator = TemporalPlateAggregator()
                self._evidence_accepted = 0
                self._evidence_rejected = 0

    def reset(self) -> None:
        with self._state_lock:
            self._aggregator = TemporalPlateAggregator()
            self._started_at = None
            self._next_analysis_at = float("-inf")
            self._frames_received = 0
            self._frames_processed = 0
            self._frames_discarded = 0
            self._detections = 0
            self._evidence_accepted = 0
            self._evidence_rejected = 0
            self._last_processing_seconds = 0.0
            self._last_error = ""

    def analyze_if_due(
        self,
        image_rgb: np.ndarray,
        now_seconds: float | None = None,
        received_frames: int = 1,
    ) -> list[PlateDetection] | None:
        """Analyze only the newest frame from a camera batch when it is due.

        ``received_frames`` includes the newest frame in ``image_rgb`` and any
        older frames already discarded by the capture layer. Keeping this
        accounting here makes backpressure visible without retaining images in
        memory.
        """
        now = self._clock() if now_seconds is None else float(now_seconds)
        batch_size = max(1, int(received_frames))
        with self._state_lock:
            self._frames_received += batch_size
            self._frames_discarded += batch_size - 1
            if self._started_at is None:
                self._started_at = now
            if now < self._next_analysis_at:
                self._frames_discarded += 1
                return None
            self._next_analysis_at = now + 1.0 / self._analyses_per_second
            frame_index = self._frames_received
            elapsed = now - self._started_at
            evidence_thresholds = self._evidence_thresholds
            roi = self._roi

        if not self._inference_lock.acquire(blocking=False):
            with self._state_lock:
                self._frames_discarded += 1
            return None

        started = self._clock()
        try:
            height, width = image_rgb.shape[:2]
            roi_x1, roi_y1, roi_x2, roi_y2 = roi.pixel_box(width, height)
            analysis_image = image_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_detections = self.recognizer.analyze_image(analysis_image)
            detections = [
                replace(
                    detection,
                    box=(
                        detection.box[0] + roi_x1,
                        detection.box[1] + roi_y1,
                        detection.box[2] + roi_x1,
                        detection.box[3] + roi_y1,
                    ),
                )
                for detection in roi_detections
            ]
            with self._state_lock:
                self._frames_processed += 1
                self._detections += len(detections)
                self._last_processing_seconds = max(0.0, self._clock() - started)
                self._last_error = ""
                for detection in detections:
                    if not evidence_thresholds.accepts(detection):
                        self._evidence_rejected += 1
                        continue
                    self._evidence_accepted += 1
                    self._aggregator.add(
                        PlateReading(
                            raw_text=detection.raw_text,
                            normalized_text=detection.normalized_text,
                            detector_confidence=detection.detector_confidence,
                            ocr_confidence=detection.ocr_confidence,
                            quality_score=detection.quality_score,
                            frame_index=frame_index,
                            time_seconds=elapsed,
                        )
                    )
            return detections
        except Exception as error:  # Keep the video stream alive and expose the error.
            with self._state_lock:
                self._last_error = f"{type(error).__name__}: {error}"
            return []
        finally:
            self._inference_lock.release()

    def snapshot(self) -> RealtimeSnapshot:
        with self._state_lock:
            return RealtimeSnapshot(
                frames_received=self._frames_received,
                frames_processed=self._frames_processed,
                frames_discarded=self._frames_discarded,
                detections=self._detections,
                evidence_accepted=self._evidence_accepted,
                evidence_rejected=self._evidence_rejected,
                last_processing_seconds=self._last_processing_seconds,
                candidates=tuple(self._aggregator.candidates()),
                decision=self._aggregator.decide(
                    self._authorized_plates,
                    min_votes=self._min_confirmations,
                    min_raw_exact_votes=1,
                ),
                last_error=self._last_error,
            )
