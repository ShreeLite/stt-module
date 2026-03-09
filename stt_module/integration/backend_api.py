from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from stt_module.compare import compare_configs
from stt_module.experiments.runner import run_experiments_from_spec
from stt_module.integration.frontend import to_frontend_payload
from stt_module.service import STTService


class BackendSTTAPI:
    """Stable facade for backend route handlers."""

    def __init__(self, service: STTService | None = None, low_confidence_threshold: float = 0.55) -> None:
        self.service = service or STTService()
        self.low_confidence_threshold = low_confidence_threshold

    def transcribe(self, audio_path: str | Path, config_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raw = self.service.transcribe(audio_path, config_overrides)
        payload = self._stable_payload(raw)
        return payload

    def compare(
        self,
        audio_path: str | Path,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = compare_configs(audio_path, config_a, config_b)
        return {
            "status": "ok",
            "mode": "comparison",
            "pipelineA": self._stable_payload(out["pipeline_a"]["full_result"]),
            "pipelineB": self._stable_payload(out["pipeline_b"]["full_result"]),
        }

    def run_experiments(self, spec_file: str | Path) -> Dict[str, Any]:
        result = run_experiments_from_spec(spec_file)
        return {"status": "ok", "mode": "experiments", "result": result}

    def _stable_payload(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        metrics = raw.get("metrics", {})
        confidence = float(raw.get("confidence", 0.0))

        warnings: list[str] = []
        if metrics.get("no_speech_detected"):
            warnings.append("No meaningful speech detected.")
        if confidence < self.low_confidence_threshold:
            warnings.append("Low confidence transcript.")

        if metrics.get("no_speech_detected"):
            status = "no_speech"
        elif confidence < self.low_confidence_threshold:
            status = "low_confidence"
        else:
            status = "ok"

        frontend_payload = to_frontend_payload(raw)

        return {
            "status": status,
            "transcript": raw.get("transcript", ""),
            "confidence": confidence,
            "latency_ms": metrics.get("total_latency_ms", 0.0),
            "audio_duration_s": metrics.get("audio_duration_s", 0.0),
            "chunk_count": metrics.get("number_of_chunks", 0),
            "chunking_strategy": metrics.get("chunking_strategy_used", "none"),
            "stage_latencies_ms": metrics.get("stage_latencies_ms", {}),
            "warnings": warnings,
            "frontend": frontend_payload,
            "raw": raw,
        }
