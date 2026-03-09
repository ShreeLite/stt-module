from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stt_module.evaluation.dataset import evaluate_dataset


class TestDatasetEvaluation(unittest.TestCase):
    def test_dataset_evaluation_outputs_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            audio = d / "a.wav"
            audio.write_bytes(b"dummy")
            mapping = d / "mapping.json"
            mapping.write_text(json.dumps({"a.wav": "hello world"}), encoding="utf-8")

            with patch("stt_module.evaluation.dataset.STTService.transcribe") as transcribe:
                transcribe.return_value = {
                    "transcript": "hello world",
                    "confidence": 0.9,
                    "metrics": {
                        "total_latency_ms": 99.0,
                        "number_of_chunks": 1,
                        "chunking_strategy_used": "none",
                    },
                }
                out = evaluate_dataset(d, mapping)

            self.assertEqual(out["summary"]["evaluated_count"], 1)
            self.assertIn("avg_wer", out["summary"])
            self.assertEqual(out["summary"]["avg_wer"], 0.0)


if __name__ == "__main__":
    unittest.main()
