from __future__ import annotations

from typing import Any, Dict


def to_frontend_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Transform service output into a frontend-friendly contract."""
    metrics = result.get("metrics", {})
    return {
        "transcript": result.get("transcript", ""),
        "confidence": result.get("confidence", 0.0),
        "status": "no_speech" if metrics.get("no_speech_detected") else "ok",
        "performance": {
            "latencyMs": metrics.get("total_latency_ms", 0.0),
            "audioDurationS": metrics.get("audio_duration_s", 0.0),
            "chunks": metrics.get("number_of_chunks", 0),
            "chunkingStrategy": metrics.get("chunking_strategy_used", "none"),
        },
        "partials": result.get("partial_transcripts", []),
        "debug": result.get("debug", {}),
    }
