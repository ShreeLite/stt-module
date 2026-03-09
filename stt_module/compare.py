from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from stt_module.service import STTService


def compare_configs(
    audio_input: str | Path,
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Run two independent pipelines on the same audio and return side-by-side results."""

    service = STTService()
    result_a = service.transcribe(audio_input, config_a)
    result_b = service.transcribe(audio_input, config_b)

    return {
        "pipeline_a": {
            "transcript": result_a["transcript"],
            "latency_ms": result_a["metrics"]["total_latency_ms"],
            "confidence": result_a["confidence"],
            "number_of_chunks": result_a["metrics"]["number_of_chunks"],
            "full_result": result_a,
        },
        "pipeline_b": {
            "transcript": result_b["transcript"],
            "latency_ms": result_b["metrics"]["total_latency_ms"],
            "confidence": result_b["confidence"],
            "number_of_chunks": result_b["metrics"]["number_of_chunks"],
            "full_result": result_b,
        },
    }
