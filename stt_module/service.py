from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from stt_module.config import STTConfig
from stt_module.logging_utils import get_logger
from stt_module.models import AudioData, STTResult
from stt_module.pipeline import STTPipeline
from stt_module.utils.audio import read_audio_input


class STTService:
    """Single-entry interface for standalone STT requests."""

    def __init__(self, base_config: STTConfig | None = None):
        self.base_config = base_config or STTConfig()
        self.logger = get_logger()
        self.pipeline = STTPipeline(self.logger)

    def transcribe(
        self,
        audio_input: str | Path | AudioData,
        config_overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        config = self.base_config.merged(config_overrides)
        audio = read_audio_input(audio_input)
        result = self.pipeline.run(audio, config)
        return self._serialize_result(result)

    def _serialize_result(self, result: STTResult) -> Dict[str, Any]:
        return {
            "transcript": result.transcript,
            "confidence": round(result.confidence, 4),
            "metrics": {
                "total_latency_ms": round(result.metrics.total_latency_ms, 3),
                "audio_duration_s": round(result.metrics.audio_duration_s, 3),
                "number_of_chunks": result.metrics.number_of_chunks,
                "model_used": result.metrics.model_used,
                "overall_confidence": round(result.metrics.overall_confidence, 4),
                "chunking_strategy_used": result.metrics.chunking_strategy_used,
                "no_speech_detected": result.metrics.no_speech_detected,
                "audio_rms": round(result.metrics.audio_rms, 6),
                "stage_latencies_ms": {
                    k: round(v, 3) for k, v in result.metrics.stage_latencies_ms.items()
                },
            },
            "partial_transcripts": [asdict(c) for c in result.partial_transcripts],
            "stage_metrics": [asdict(m) for m in result.stage_metrics],
            "debug": {
                "vad_segments": [asdict(seg) for seg in result.debug.vad_segments],
                "chunk_boundaries": [asdict(seg) for seg in result.debug.chunk_boundaries],
                "waveform_envelope": result.debug.waveform_envelope,
            },
        }
