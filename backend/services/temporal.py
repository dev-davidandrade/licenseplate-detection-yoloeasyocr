from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from backend.services.plate_utils import clean_text, is_valid_plate, normalize_plate_text


@dataclass(frozen=True)
class PlateReading:
    raw_text: str
    normalized_text: str
    detector_confidence: float
    ocr_confidence: float
    quality_score: float
    frame_index: int = 0
    time_seconds: float = 0.0

    @property
    def weight(self) -> float:
        return (
            max(self.detector_confidence, 0.01)
            * max(self.ocr_confidence, 0.01)
            * (0.5 + 0.5 * self.quality_score)
        )


@dataclass(frozen=True)
class PlateCandidate:
    plate: str
    votes: int
    raw_exact_votes: int
    weighted_score: float
    best_confidence: float


@dataclass(frozen=True)
class AuthorizationDecision:
    authorized: bool
    plate: str
    reason: str
    votes: int
    raw_exact_votes: int


def parse_authorized_plates(raw: str) -> set[str]:
    candidates = raw.replace(",", "\n").replace(";", "\n").splitlines()
    return {
        cleaned
        for candidate in candidates
        if is_valid_plate(cleaned := clean_text(candidate))
    }


class TemporalPlateAggregator:
    def __init__(self) -> None:
        self._readings: list[PlateReading] = []

    @property
    def readings(self) -> tuple[PlateReading, ...]:
        return tuple(self._readings)

    def add(self, reading: PlateReading) -> None:
        self._readings.append(reading)

    def candidates(self) -> list[PlateCandidate]:
        grouped: dict[str, list[PlateReading]] = defaultdict(list)
        for reading in self._readings:
            plate = normalize_plate_text(reading.normalized_text or reading.raw_text)
            if is_valid_plate(plate):
                grouped[plate].append(reading)

        candidates = []
        for plate, readings in grouped.items():
            candidates.append(
                PlateCandidate(
                    plate=plate,
                    votes=len(readings),
                    raw_exact_votes=sum(
                        clean_text(reading.raw_text) == plate for reading in readings
                    ),
                    weighted_score=round(sum(reading.weight for reading in readings), 6),
                    best_confidence=round(
                        max(
                            reading.detector_confidence * reading.ocr_confidence
                            for reading in readings
                        ),
                        6,
                    ),
                )
            )
        return sorted(
            candidates,
            key=lambda item: (
                item.votes,
                item.raw_exact_votes,
                item.weighted_score,
            ),
            reverse=True,
        )

    def decide(
        self,
        authorized_plates: set[str],
        min_votes: int = 2,
        min_raw_exact_votes: int = 1,
        max_gap_seconds: float = 3.0,
    ) -> AuthorizationDecision:
        if not authorized_plates:
            return AuthorizationDecision(False, "", "lista_vazia", 0, 0)

        matching_readings: dict[str, list[PlateReading]] = defaultdict(list)
        for reading in self._readings:
            plate = normalize_plate_text(reading.normalized_text or reading.raw_text)
            if plate in authorized_plates:
                matching_readings[plate].append(reading)
        if not matching_readings:
            return AuthorizationDecision(False, "", "sem_correspondencia", 0, 0)

        passage_candidates = []
        for plate, readings in matching_readings.items():
            ordered = sorted(readings, key=lambda reading: reading.time_seconds)
            passages: list[list[PlateReading]] = []
            for reading in ordered:
                if (
                    not passages
                    or reading.time_seconds - passages[-1][-1].time_seconds
                    > max_gap_seconds
                ):
                    passages.append([reading])
                else:
                    passages[-1].append(reading)
            for passage in passages:
                passage_candidates.append(
                    PlateCandidate(
                        plate=plate,
                        votes=len(passage),
                        raw_exact_votes=sum(
                            clean_text(reading.raw_text) == plate for reading in passage
                        ),
                        weighted_score=round(
                            sum(reading.weight for reading in passage), 6
                        ),
                        best_confidence=round(
                            max(
                                reading.detector_confidence * reading.ocr_confidence
                                for reading in passage
                            ),
                            6,
                        ),
                    )
                )
        best = max(
            passage_candidates,
            key=lambda item: (
                item.votes,
                item.raw_exact_votes,
                item.weighted_score,
            ),
        )
        if best.votes < min_votes:
            return AuthorizationDecision(
                False,
                best.plate,
                "confirmacoes_insuficientes",
                best.votes,
                best.raw_exact_votes,
            )
        if best.raw_exact_votes < min_raw_exact_votes:
            return AuthorizationDecision(
                False,
                best.plate,
                "sem_leitura_exata",
                best.votes,
                best.raw_exact_votes,
            )
        return AuthorizationDecision(
            True,
            best.plate,
            "placa_confirmada",
            best.votes,
            best.raw_exact_votes,
        )
