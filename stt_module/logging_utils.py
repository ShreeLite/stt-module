from __future__ import annotations

import logging
from typing import Any, Dict


def get_logger(name: str = "stt_module") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_pipeline_start(logger: logging.Logger, config: Dict[str, Any], model_name: str) -> None:
    logger.info(
        "pipeline_start enabled_modules=%s model=%s",
        {
            "preprocessing": config.get("enable_preprocessing"),
            "noise_reduction": config.get("enable_noise_reduction"),
            "vad": config.get("enable_vad"),
            "chunking": config.get("enable_chunking"),
            "postprocessing": config.get("enable_postprocessing"),
            "confidence_filtering": config.get("enable_confidence_filtering"),
        },
        model_name,
    )


def log_pipeline_end(
    logger: logging.Logger,
    total_latency_ms: float,
    chunk_count: int,
    confidence: float,
) -> None:
    logger.info(
        "pipeline_end latency_ms=%.2f chunks=%d confidence=%.3f",
        total_latency_ms,
        chunk_count,
        confidence,
    )


def log_chunking_strategy(logger: logging.Logger, strategy: str, reason: str) -> None:
    logger.info("chunking_strategy strategy=%s reason=%s", strategy, reason)


def log_silence_decision(logger: logging.Logger, no_speech_detected: bool, audio_rms: float) -> None:
    logger.info(
        "silence_detection no_speech=%s audio_rms=%.6f",
        no_speech_detected,
        audio_rms,
    )
