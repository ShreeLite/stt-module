from __future__ import annotations

import unittest
from unittest.mock import patch

from stt_module.compare import compare_configs


class TestCompare(unittest.TestCase):
    def test_compare_returns_dual_results(self) -> None:
        with patch("stt_module.compare.STTService.transcribe") as transcribe:
            transcribe.side_effect = [
                {
                    "transcript": "pipeline a",
                    "confidence": 0.71,
                    "metrics": {"total_latency_ms": 101.0, "number_of_chunks": 3},
                },
                {
                    "transcript": "pipeline b",
                    "confidence": 0.88,
                    "metrics": {"total_latency_ms": 95.0, "number_of_chunks": 2},
                },
            ]

            output = compare_configs("dummy.wav", {"enable_vad": True}, {"enable_vad": False})

        self.assertEqual(output["pipeline_a"]["transcript"], "pipeline a")
        self.assertEqual(output["pipeline_b"]["transcript"], "pipeline b")
        self.assertEqual(output["pipeline_a"]["number_of_chunks"], 3)
        self.assertEqual(output["pipeline_b"]["number_of_chunks"], 2)


if __name__ == "__main__":
    unittest.main()
