from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from stt_module.evaluation.metrics import compute_wer_cer
from stt_module.integration.backend_api import BackendSTTAPI


api = BackendSTTAPI()


def handle_transcribe_request(audio_file_path: str, config: Dict[str, Any] | None = None) -> Tuple[int, Dict[str, Any]]:
    """Reference route handler for transcription requests."""
    result = api.transcribe(audio_file_path, config)
    return 200, result


def handle_compare_request(
    audio_file_path: str,
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    """Reference route handler for comparison mode."""
    result = api.compare(audio_file_path, config_a, config_b)
    return 200, result


def handle_experiments_request(spec_file_path: str) -> Tuple[int, Dict[str, Any]]:
    """Reference route handler for experiment batches from spec file."""
    result = api.run_experiments(spec_file_path)
    return 200, result


def handle_evaluate_request(audio_file_path: str, hypothesis: str, ground_truth: str) -> Tuple[int, Dict[str, Any]]:
    """Reference route handler for transcript evaluation against ground truth."""
    quality = compute_wer_cer(reference=ground_truth, hypothesis=hypothesis)
    return 200, {
        "status": "ok",
        "audio": str(Path(audio_file_path).name),
        "wer": quality["wer"],
        "cer": quality["cer"],
    }
