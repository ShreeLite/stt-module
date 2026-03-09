from __future__ import annotations

import unittest

import numpy as np

from stt_module.config import STTConfig
from stt_module.models import AudioData
from stt_module.stages.preprocessing import Preprocessor


class TestPreprocessing(unittest.TestCase):
    def test_preprocessing_converts_to_mono_and_target_rate(self) -> None:
        # Simulate stereo audio at 8kHz; preprocessing should normalize to mono 16kHz.
        left = np.sin(2 * np.pi * 220 * np.arange(8000) / 8000).astype(np.float32)
        right = np.sin(2 * np.pi * 330 * np.arange(8000) / 8000).astype(np.float32)
        stereo = np.stack([left, right], axis=1)

        audio = AudioData(
            samples=stereo,
            sample_rate_hz=8000,
            channels=2,
            sample_width_bytes=2,
        )

        cfg = STTConfig(target_sample_rate_hz=16000, target_channels=1)
        processed = Preprocessor().run(audio, cfg)

        self.assertEqual(processed.channels, 1)
        self.assertEqual(processed.sample_rate_hz, 16000)
        self.assertEqual(processed.samples.ndim, 1)
        self.assertGreater(processed.samples.shape[0], audio.samples.shape[0])


if __name__ == "__main__":
    unittest.main()
