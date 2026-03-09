from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from stt_module.compare import compare_configs
from stt_module.service import STTService


class TestVoiceSampleIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audio_path = Path(__file__).resolve().parents[1] / "voice-sample.wav"
        if not cls.audio_path.exists():
            raise FileNotFoundError(f"Missing integration fixture: {cls.audio_path}")

        cls.service = STTService()
        cls._cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def _run(cls, overrides: dict[str, Any]) -> dict[str, Any]:
        key = json.dumps(overrides, sort_keys=True)
        if key not in cls._cache:
            cls._cache[key] = cls.service.transcribe(cls.audio_path, overrides)
        return cls._cache[key]

    def test_baseline_schema_and_metrics(self) -> None:
        result = self._run(
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": False,
                "enable_chunking": False,
                "enable_noise_reduction": False,
                "enable_postprocessing": True,
                "enable_confidence_filtering": False,
            }
        )

        self.assertIn("transcript", result)
        self.assertIn("confidence", result)
        self.assertIn("metrics", result)
        self.assertIn("partial_transcripts", result)
        self.assertIn("stage_metrics", result)
        self.assertIn("debug", result)

        metrics = result["metrics"]
        self.assertGreater(metrics["total_latency_ms"], 0.0)
        self.assertGreater(metrics["audio_duration_s"], 0.0)
        self.assertEqual(metrics["model_used"], "tiny")
        self.assertEqual(metrics["number_of_chunks"], 1)
        self.assertEqual(len(result["partial_transcripts"]), 1)

    def test_noise_reduction_stage_executes(self) -> None:
        result = self._run(
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": False,
                "enable_chunking": False,
                "enable_noise_reduction": True,
                "enable_postprocessing": True,
            }
        )

        stage_latencies = result["metrics"]["stage_latencies_ms"]
        self.assertIn("noise_reduction", stage_latencies)
        self.assertGreaterEqual(stage_latencies["noise_reduction"], 0.0)

    def test_vad_chunking_streaming_outputs(self) -> None:
        result = self._run(
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": True,
                "enable_chunking": True,
                "chunking_mode": "vad",
                "vad_frame_ms": 30,
                "enable_debug_visualization": True,
                "enable_postprocessing": True,
            }
        )

        chunk_count = result["metrics"]["number_of_chunks"]
        partials = result["partial_transcripts"]
        chunk_boundaries = result["debug"]["chunk_boundaries"]

        # VAD may produce 0 chunks for low-speech files; still validates streaming contract.
        self.assertEqual(len(partials), chunk_count)
        self.assertEqual(len(chunk_boundaries), chunk_count)
        self.assertIn("vad", result["metrics"]["stage_latencies_ms"])
        self.assertIn("chunking", result["metrics"]["stage_latencies_ms"])
        self.assertIn("vad_segments", result["debug"])

    def test_fixed_chunking_with_overlap_generates_multiple_partials(self) -> None:
        result = self._run(
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": False,
                "enable_chunking": True,
                "chunking_mode": "fixed",
                "fixed_chunk_duration_s": 3.0,
                "fixed_chunk_overlap_s": 0.75,
                "short_audio_no_chunk_threshold_s": 0.1,
                "enable_postprocessing": True,
            }
        )

        chunk_count = result["metrics"]["number_of_chunks"]
        self.assertGreaterEqual(chunk_count, 2)
        self.assertEqual(chunk_count, len(result["partial_transcripts"]))

        boundaries = result["debug"]["chunk_boundaries"]
        if len(boundaries) >= 2:
            self.assertLess(boundaries[1]["start_s"], boundaries[0]["end_s"])

    def test_confidence_filtering_stage_executes(self) -> None:
        result = self._run(
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": False,
                "enable_chunking": False,
                "enable_confidence_filtering": True,
                "confidence_threshold": 0.99,
                "enable_postprocessing": True,
            }
        )

        stage_latencies = result["metrics"]["stage_latencies_ms"]
        self.assertIn("confidence_filter", stage_latencies)
        self.assertGreaterEqual(stage_latencies["confidence_filter"], 0.0)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_postprocessing_toggle_changes_stage_execution(self) -> None:
        with_post = self._run(
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": False,
                "enable_chunking": False,
                "enable_postprocessing": True,
            }
        )
        without_post = self._run(
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": False,
                "enable_chunking": False,
                "enable_postprocessing": False,
            }
        )

        self.assertIn("postprocessing", with_post["metrics"]["stage_latencies_ms"])
        self.assertNotIn("postprocessing", without_post["metrics"]["stage_latencies_ms"])

    def test_debug_waveform_population_toggle(self) -> None:
        with_debug = self._run(
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": False,
                "enable_chunking": False,
                "enable_debug_visualization": True,
            }
        )
        without_debug = self._run(
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": False,
                "enable_chunking": False,
                "enable_debug_visualization": False,
            }
        )

        self.assertGreater(len(with_debug["debug"]["waveform_envelope"]), 0)
        self.assertEqual(without_debug["debug"]["waveform_envelope"], [])

    def test_comparison_mode_on_voice_sample(self) -> None:
        result = compare_configs(
            self.audio_path,
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": True,
                "enable_chunking": True,
                "chunking_mode": "vad",
            },
            {
                "model_name": "tiny",
                "model_device": "cpu",
                "enable_vad": False,
                "enable_chunking": False,
            },
        )

        self.assertIn("pipeline_a", result)
        self.assertIn("pipeline_b", result)

        for key in ("pipeline_a", "pipeline_b"):
            self.assertIn("transcript", result[key])
            self.assertIn("latency_ms", result[key])
            self.assertIn("confidence", result[key])
            self.assertIn("number_of_chunks", result[key])
            self.assertGreaterEqual(result[key]["latency_ms"], 0.0)
            self.assertGreaterEqual(result[key]["confidence"], 0.0)
            self.assertLessEqual(result[key]["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
