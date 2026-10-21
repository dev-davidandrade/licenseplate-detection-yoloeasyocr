from __future__ import annotations

import threading
import time
from dataclasses import dataclass
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
    detections: int
    last_processing_seconds: float
    candidates: tuple[PlateCandidate, ...]
    decision: AuthorizationDecision
    last_error: str


class RealtimePlateSession:
    """Thread-safe state shared by Streamlit and the WebRTC callback."""

    def __init__(
        self,
        recognizer: PlateRecognizer,
        analyses_per_second: float = 2.0,
        authorized_plates: set[str] | None = None,
        min_confirmations: int = 2,
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
        self._detections = 0
        self._last_processing_seconds = 0.0
        self._last_error = ""
        self.configure(
            analyses_per_second=analyses_per_second,
            authorized_plates=authorized_plates or set(),
            min_confirmations=min_confirmations,
        )

    def configure(
        self,
        analyses_per_second: float,
        authorized_plates: set[str],
        min_confirmations: int,
    ) -> None:
        with self._state_lock:
            self._analyses_per_second = max(0.1, float(analyses_per_second))
            self._authorized_plates = set(authorized_plates)
            self._min_confirmations = max(1, int(min_confirmations))

    def reset(self) -> None:
        with self._state_lock:
            self._aggregator = TemporalPlateAggregator()
            self._started_at = None
            self._next_analysis_at = float("-inf")
            self._frames_received = 0
            self._frames_processed = 0
            self._detections = 0
            self._last_processing_seconds = 0.0
            self._last_error = ""

    def analyze_if_due(
        self,
        image_rgb: np.ndarray,
        now_seconds: float | None = None,
    ) -> list[PlateDetection] | None:
        now = self._clock() if now_seconds is None else float(now_seconds)
        with self._state_lock:
            self._frames_received += 1
            if self._started_at is None:
                self._started_at = now
            if now < self._next_analysis_at:
                return None
            self._next_analysis_at = now + 1.0 / self._analyses_per_second
            frame_index = self._frames_received
            elapsed = now - self._started_at

        if not self._inference_lock.acquire(blocking=False):
            return None

        started = self._clock()
        try:
            detections = self.recognizer.analyze_image(image_rgb)
            with self._state_lock:
                self._frames_processed += 1
                self._detections += len(detections)
                self._last_processing_seconds = max(0.0, self._clock() - started)
                self._last_error = ""
                for detection in detections:
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
                detections=self._detections,
                last_processing_seconds=self._last_processing_seconds,
                candidates=tuple(self._aggregator.candidates()),
                decision=self._aggregator.decide(
                    self._authorized_plates,
                    min_votes=self._min_confirmations,
                    min_raw_exact_votes=1,
                ),
                last_error=self._last_error,
            )
