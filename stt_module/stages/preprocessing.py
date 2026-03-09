from __future__ import annotations

from stt_module.config import STTConfig
from stt_module.models import AudioData
from stt_module.utils.audio import ensure_format


class Preprocessor:
    name = "preprocessing"

    def run(self, audio: AudioData, config: STTConfig) -> AudioData:
        return ensure_format(
            audio,
            target_sample_rate_hz=config.target_sample_rate_hz,
            target_channels=config.target_channels,
        )
