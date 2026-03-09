from __future__ import annotations

import unittest

import numpy as np

from stt_module.config import STTConfig
from stt_module.models import AudioData, SpeechSegment
from stt_module.stages.chunking import Chunker


class TestChunking(unittest.TestCase):
    def setUp(self) -> None:
        self.audio = AudioData(
            samples=np.zeros(16000 * 10, dtype=np.float32),
            sample_rate_hz=16000,
            channels=1,
            sample_width_bytes=2,
        )
        self.chunker = Chunker()

    def test_vad_chunking_respects_segments(self) -> None:
        cfg = STTConfig(chunking_mode="vad", short_audio_no_chunk_threshold_s=0.1)
        segments = [SpeechSegment(0.2, 2.2), SpeechSegment(2.3, 4.5)]
        chunks = self.chunker.create_chunks(self.audio, cfg, segments)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertGreater(chunks[0].duration_s, 2.0)

    def test_vad_mode_without_segments_returns_empty(self) -> None:
        cfg = STTConfig(chunking_mode="vad", short_audio_no_chunk_threshold_s=0.1)
        chunks = self.chunker.create_chunks(self.audio, cfg, [])
        self.assertEqual(chunks, [])

    def test_fixed_chunking_uses_overlap(self) -> None:
        cfg = STTConfig(
            chunking_mode="fixed",
            fixed_chunk_duration_s=4.0,
            fixed_chunk_overlap_s=1.0,
            short_audio_no_chunk_threshold_s=0.1,
        )
        chunks = self.chunker.create_chunks(self.audio, cfg, None)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertLess(chunks[1].start_s, chunks[0].end_s)


if __name__ == "__main__":
    unittest.main()
