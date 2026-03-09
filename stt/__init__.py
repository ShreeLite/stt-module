"""Compatibility import package for backend integration."""

from stt_module import (
    STTConfig,
    STTService,
    BackendSTTAPI,
    compare_configs,
    compute_wer_cer,
    evaluate_dataset,
)

__all__ = [
    "STTConfig",
    "STTService",
    "BackendSTTAPI",
    "compare_configs",
    "compute_wer_cer",
    "evaluate_dataset",
]
