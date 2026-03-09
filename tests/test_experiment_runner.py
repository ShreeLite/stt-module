from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stt_module.experiments.runner import ExperimentConfig, ExperimentRunner, load_experiment_spec


class TestExperimentRunner(unittest.TestCase):
    def test_runner_collects_results_and_summary(self) -> None:
        runner = ExperimentRunner()
        experiments = [
            ExperimentConfig("baseline", {"enable_vad": False}),
            ExperimentConfig("vad", {"enable_vad": True}),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "a.wav"
            audio.write_bytes(b"dummy")

            with patch.object(runner.service, "transcribe") as transcribe:
                transcribe.side_effect = [
                    {
                        "transcript": "hello",
                        "confidence": 0.7,
                        "metrics": {
                            "total_latency_ms": 100.0,
                            "number_of_chunks": 1,
                            "stage_latencies_ms": {"recognition": 99.0},
                            "model_used": "tiny",
                            "chunking_strategy_used": "none",
                            "no_speech_detected": False,
                        },
                    },
                    {
                        "transcript": "hello",
                        "confidence": 0.75,
                        "metrics": {
                            "total_latency_ms": 120.0,
                            "number_of_chunks": 3,
                            "stage_latencies_ms": {"recognition": 110.0},
                            "model_used": "tiny",
                            "chunking_strategy_used": "vad",
                            "no_speech_detected": False,
                        },
                    },
                ]

                result = runner.run(audio_input=audio, experiments=experiments)

            self.assertEqual(len(result["results"]), 2)
            self.assertIn("avg_latency_ms", result["summary"])

            out_path = Path(tmpdir) / "results.json"
            runner.write_results(result, out_path)
            self.assertTrue(out_path.exists())

    def test_load_experiment_spec_json_and_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            json_spec = d / "spec.json"
            json_spec.write_text(
                json.dumps({"audio_input": "voice.wav", "experiments": [{"name": "a", "overrides": {}}]}),
                encoding="utf-8",
            )

            yaml_spec = d / "spec.yaml"
            yaml_spec.write_text(
                "audio_input: voice.wav\nexperiments:\n  - name: a\n    overrides: {}\n",
                encoding="utf-8",
            )

            self.assertIn("audio_input", load_experiment_spec(json_spec))
            self.assertIn("audio_input", load_experiment_spec(yaml_spec))


if __name__ == "__main__":
    unittest.main()
