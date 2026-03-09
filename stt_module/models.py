from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass(slots=True)
class AudioData:
    samples: np.ndarray
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int

    @property
    def duration_s(self) -> float:
        if self.sample_rate_hz <= 0:
            return 0.0
        return float(self.samples.shape[0]) / float(self.sample_rate_hz)


@dataclass(slots=True)
class SpeechSegment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(slots=True)
class Chunk:
    index: int
    start_s: float
    end_s: float
    samples: np.ndarray

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(slots=True)
class ChunkTranscript:
    chunk_index: int
    text: str
    confidence: float
    start_s: float
    end_s: float


@dataclass(slots=True)
class StageMetric:
    stage_name: str
    latency_ms: float
    skipped: bool = False


@dataclass(slots=True)
class PipelineMetrics:
    total_latency_ms: float
    audio_duration_s: float
    number_of_chunks: int
    model_used: str
    overall_confidence: float
    chunking_strategy_used: str = "none"
    no_speech_detected: bool = False
    audio_rms: float = 0.0
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DebugInfo:
    vad_segments: List[SpeechSegment] = field(default_factory=list)
    chunk_boundaries: List[SpeechSegment] = field(default_factory=list)
    waveform_envelope: List[float] = field(default_factory=list)


@dataclass(slots=True)
class STTResult:
    transcript: str
    confidence: float
    metrics: PipelineMetrics
    partial_transcripts: List[ChunkTranscript]
    stage_metrics: List[StageMetric]
    debug: DebugInfo = field(default_factory=DebugInfo)
