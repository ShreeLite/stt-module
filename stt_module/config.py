from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal

ChunkingMode = Literal["vad", "fixed"]
ChunkingPolicy = Literal["manual", "auto"]
ModelName = Literal[
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v3",
]


@dataclass(slots=True)
class STTConfig:
    """Runtime-configurable settings for the STT pipeline."""

    enable_preprocessing: bool = True
    enable_noise_reduction: bool = False
    enable_vad: bool = True
    enable_chunking: bool = True
    chunking_policy: ChunkingPolicy = "manual"
    enable_postprocessing: bool = True
    enable_confidence_filtering: bool = False

    model_name: ModelName = "small"
    model_device: Literal["cpu", "cuda"] = "cpu"
    model_compute_type: str = "int8"
    model_beam_size: int = 1
    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    condition_on_previous_text: bool = False

    chunking_mode: ChunkingMode = "vad"
    fixed_chunk_duration_s: float = 8.0
    fixed_chunk_overlap_s: float = 0.75
    min_chunk_duration_s: float = 2.5
    short_audio_no_chunk_threshold_s: float = 4.0
    auto_chunking_duration_threshold_s: float = 28.0

    vad_sensitivity: float = 0.55
    vad_frame_ms: int = 30
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 200
    vad_group_pause_s: float = 0.45
    vad_max_chunk_s: float = 12.0

    confidence_threshold: float = 0.45
    enable_silence_detection: bool = True
    silence_rms_threshold: float = 0.004

    target_sample_rate_hz: int = 16000
    target_channels: int = 1
    sample_width_bytes: int = 2

    enable_debug_visualization: bool = False

    def __post_init__(self) -> None:
        if self.fixed_chunk_duration_s <= 0:
            raise ValueError("fixed_chunk_duration_s must be > 0")
        if self.fixed_chunk_overlap_s < 0:
            raise ValueError("fixed_chunk_overlap_s must be >= 0")
        if self.fixed_chunk_overlap_s >= self.fixed_chunk_duration_s:
            raise ValueError("fixed_chunk_overlap_s must be less than fixed_chunk_duration_s")
        if self.min_chunk_duration_s <= 0:
            raise ValueError("min_chunk_duration_s must be > 0")
        if self.auto_chunking_duration_threshold_s <= 0:
            raise ValueError("auto_chunking_duration_threshold_s must be > 0")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.silence_rms_threshold < 0.0:
            raise ValueError("silence_rms_threshold must be >= 0")
        if not 0.0 <= self.vad_sensitivity <= 1.0:
            raise ValueError("vad_sensitivity must be between 0 and 1")
        if self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be > 0")
        if self.target_channels != 1:
            raise ValueError("target_channels must be 1 for speech recognition")
        if self.model_beam_size < 1:
            raise ValueError("model_beam_size must be >= 1")

    def merged(self, overrides: Dict[str, Any] | None) -> "STTConfig":
        if not overrides:
            return self
        current = asdict(self)
        current.update(overrides)
        return STTConfig(**current)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
