from __future__ import annotations

import noisereduce as nr
import numpy as np

from stt_module.config import STTConfig
from stt_module.models import AudioData


class NoiseReducer:
    name = "noise_reduction"

    def run(self, audio: AudioData, config: STTConfig) -> AudioData:
        reduced = nr.reduce_noise(
            y=np.asarray(audio.samples, dtype=np.float32),
            sr=audio.sample_rate_hz,
            stationary=False,
            prop_decrease=0.85,
        ).astype(np.float32)
        return AudioData(
            samples=reduced,
            sample_rate_hz=audio.sample_rate_hz,
            channels=audio.channels,
            sample_width_bytes=audio.sample_width_bytes,
        )
