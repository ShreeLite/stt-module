from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts.verify_stt_environment import verify_environment


class TestEnvironmentVerifier(unittest.TestCase):
    def test_verify_environment_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "sample.wav"
            data = (0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
            sf.write(audio, data, 16000)

            with patch("scripts.verify_stt_environment.SpeechRecognizer._get_model") as get_model:
                get_model.return_value = object()
                out = verify_environment(audio, model_name="tiny", device="cpu")

            self.assertTrue(out["all_ok"])
            self.assertTrue(out["audio_decode_ok"])
            self.assertTrue(out["model_init_ok"])


if __name__ == "__main__":
    unittest.main()
