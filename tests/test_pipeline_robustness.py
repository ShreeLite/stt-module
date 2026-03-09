from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from stt_module.config import STTConfig
from stt_module.models import AudioData, ChunkTranscript, SpeechSegment
from stt_module.pipeline import STTPipeline


class DummyLogger:
    def info(self, *args, **kwargs):
        return None


class TestPipelineRobustness(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = STTPipeline(DummyLogger())

    def test_auto_chunking_disables_for_short_audio(self) -> None:
        audio = AudioData(np.random.randn(16000 * 5).astype(np.float32), 16000, 1, 2)
        cfg = STTConfig(
            chunking_policy="auto",
            auto_chunking_duration_threshold_s=28.0,
            enable_vad=True,
            model_name="tiny",
        )

        with patch.object(
            self.pipeline.recognizer,
            "transcribe_chunk",
            return_value=ChunkTranscript(0, "short audio", 0.8, 0.0, 5.0),
        ):
            result = self.pipeline.run(audio, cfg)

        self.assertEqual(result.metrics.chunking_strategy_used, "none")
        self.assertEqual(result.metrics.number_of_chunks, 1)

    def test_auto_chunking_prefers_vad_for_long_audio(self) -> None:
        audio = AudioData(np.random.randn(16000 * 40).astype(np.float32), 16000, 1, 2)
        cfg = STTConfig(
            chunking_policy="auto",
            auto_chunking_duration_threshold_s=25.0,
            enable_vad=True,
            model_name="tiny",
        )

        fake_segments = [SpeechSegment(0.0, 8.0), SpeechSegment(8.3, 18.0), SpeechSegment(18.4, 30.0)]
        with patch.object(self.pipeline.vad, "detect", return_value=fake_segments), patch.object(
            self.pipeline.recognizer,
            "transcribe_chunk",
            side_effect=[
                ChunkTranscript(0, "a", 0.7, 0.0, 8.0),
                ChunkTranscript(1, "b", 0.7, 8.3, 18.0),
                ChunkTranscript(2, "c", 0.7, 18.4, 30.0),
            ],
        ):
            result = self.pipeline.run(audio, cfg)

        self.assertEqual(result.metrics.chunking_strategy_used, "vad")
        self.assertGreaterEqual(result.metrics.number_of_chunks, 1)

    def test_auto_chunking_falls_back_to_fixed_when_vad_unavailable(self) -> None:
        audio = AudioData(np.random.randn(16000 * 35).astype(np.float32), 16000, 1, 2)
        cfg = STTConfig(
            chunking_policy="auto",
            auto_chunking_duration_threshold_s=25.0,
            enable_vad=False,
            model_name="tiny",
        )

        with patch.object(
            self.pipeline.recognizer,
            "transcribe_chunk",
            return_value=ChunkTranscript(0, "x", 0.6, 0.0, 3.0),
        ):
            result = self.pipeline.run(audio, cfg)

        self.assertEqual(result.metrics.chunking_strategy_used, "fixed")
        self.assertGreater(result.metrics.number_of_chunks, 1)

    def test_silence_detection_skips_recognition(self) -> None:
        audio = AudioData(np.zeros(16000 * 10, dtype=np.float32), 16000, 1, 2)
        cfg = STTConfig(
            enable_silence_detection=True,
            enable_vad=True,
            silence_rms_threshold=0.001,
            model_name="tiny",
        )

        with patch.object(self.pipeline.vad, "detect", return_value=[]), patch.object(
            self.pipeline.recognizer, "transcribe_chunk"
        ) as mock_tx:
            result = self.pipeline.run(audio, cfg)

        mock_tx.assert_not_called()
        self.assertTrue(result.metrics.no_speech_detected)
        self.assertEqual(result.transcript, "")


if __name__ == "__main__":
    unittest.main()
