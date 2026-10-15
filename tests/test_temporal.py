import unittest

from backend.services.temporal import (
    PlateReading,
    TemporalPlateAggregator,
    parse_authorized_plates,
)


def reading(raw: str, normalized: str, time_seconds: float = 0.0) -> PlateReading:
    return PlateReading(
        raw_text=raw,
        normalized_text=normalized,
        detector_confidence=0.9,
        ocr_confidence=0.8,
        quality_score=0.7,
        time_seconds=time_seconds,
    )


class TemporalPlateAggregatorTests(unittest.TestCase):
    def test_parses_only_supported_brazilian_plates(self):
        plates = parse_authorized_plates("ABC1234, BRA2E19\ninválida")

        self.assertEqual(plates, {"ABC1234", "BRA2E19"})

    def test_authorizes_after_repeated_evidence_and_one_raw_exact_read(self):
        aggregator = TemporalPlateAggregator()
        aggregator.add(reading("CXT0522", "CXT0522"))
        aggregator.add(reading("CXTOSZZ", "CXT0522"))

        decision = aggregator.decide({"CXT0522"})

        self.assertTrue(decision.authorized)
        self.assertEqual(decision.plate, "CXT0522")
        self.assertEqual(decision.votes, 2)
        self.assertEqual(decision.raw_exact_votes, 1)

    def test_does_not_authorize_a_single_read(self):
        aggregator = TemporalPlateAggregator()
        aggregator.add(reading("CXT0522", "CXT0522"))

        decision = aggregator.decide({"CXT0522"})

        self.assertFalse(decision.authorized)
        self.assertEqual(decision.reason, "confirmacoes_insuficientes")

    def test_does_not_authorize_normalized_guesses_without_raw_exact_read(self):
        aggregator = TemporalPlateAggregator()
        aggregator.add(reading("CXI0SZZ", "CXI0522"))
        aggregator.add(reading("CXI0SZZ", "CXI0522"))

        decision = aggregator.decide({"CXI0522"})

        self.assertFalse(decision.authorized)
        self.assertEqual(decision.reason, "sem_leitura_exata")

    def test_ignores_non_authorized_candidates(self):
        aggregator = TemporalPlateAggregator()
        aggregator.add(reading("ABC1234", "ABC1234"))
        aggregator.add(reading("ABC1234", "ABC1234"))

        decision = aggregator.decide({"CXT0522"})

        self.assertFalse(decision.authorized)
        self.assertEqual(decision.reason, "sem_correspondencia")

    def test_does_not_combine_confirmations_from_separate_passages(self):
        aggregator = TemporalPlateAggregator()
        aggregator.add(reading("CXT0522", "CXT0522", time_seconds=1.0))
        aggregator.add(reading("CXT0522", "CXT0522", time_seconds=10.0))

        decision = aggregator.decide({"CXT0522"}, max_gap_seconds=3.0)

        self.assertFalse(decision.authorized)
        self.assertEqual(decision.reason, "confirmacoes_insuficientes")
        self.assertEqual(decision.votes, 1)


if __name__ == "__main__":
    unittest.main()
