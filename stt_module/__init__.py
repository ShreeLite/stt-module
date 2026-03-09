"""Standalone modular Speech-to-Text experimentation module."""

from .config import STTConfig
from .service import STTService
from .compare import compare_configs

__all__ = ["STTConfig", "STTService", "compare_configs"]
