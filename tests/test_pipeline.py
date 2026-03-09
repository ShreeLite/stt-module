from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from stt_module.config import STTConfig
from stt_module.models import AudioData, ChunkTranscript
from stt_module.pipeline import STTPipeline


class DummyLogger:
    def info(self, *args, **kwargs):
        return None


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.audio = AudioData(
            samples=np.random.default_rng(42).normal(0, 0.02, 16000 * 5).astype(np.float32),
            sample_rate_hz=16000,
            channels=1,
            sample_width_bytes=2,
        )

    def test_pipeline_emits_metrics_and_partials(self) -> None:
        pipeline = STTPipeline(DummyLogger())
        cfg = STTConfig(
            enable_vad=False,
            enable_chunking=False,
            enable_noise_reduction=False,
            enable_confidence_filtering=False,
            model_name="tiny",
        )

        fake_chunk_result = ChunkTranscript(
            chunk_index=0,
            text="hello world",
            confidence=0.9,
            start_s=0.0,
            end_s=5.0,
        )

        with patch.object(pipeline.recognizer, "transcribe_chunk", return_value=fake_chunk_result):
            result = pipeline.run(self.audio, cfg)

        self.assertEqual(result.transcript, "Hello world.")
        self.assertEqual(len(result.partial_transcripts), 1)
        self.assertGreater(result.metrics.total_latency_ms, 0.0)
        self.assertIn("recognition", result.metrics.stage_latencies_ms)

    def test_confidence_filter_can_remove_low_confidence_chunks(self) -> None:
        pipeline = STTPipeline(DummyLogger())
        cfg = STTConfig(
            enable_vad=False,
            enable_chunking=False,
            enable_confidence_filtering=True,
            confidence_threshold=0.95,
            model_name="tiny",
        )

        low_conf = ChunkTranscript(
            chunk_index=0,
            text="low confidence",
            confidence=0.5,
            start_s=0.0,
            end_s=5.0,
        )

        with patch.object(pipeline.recognizer, "transcribe_chunk", return_value=low_conf):
            result = pipeline.run(self.audio, cfg)

        self.assertEqual(result.transcript, "")
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
