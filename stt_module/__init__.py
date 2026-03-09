"""Standalone modular Speech-to-Text experimentation module."""

from .config import STTConfig
from .service import STTService
from .compare import compare_configs
from .evaluation.metrics import compute_wer_cer
from .evaluation.dataset import evaluate_dataset
from .experiments.runner import ExperimentRunner, run_experiments_from_spec
from .integration.backend_api import BackendSTTAPI

__all__ = [
	"STTConfig",
	"STTService",
	"compare_configs",
	"compute_wer_cer",
	"evaluate_dataset",
	"ExperimentRunner",
	"run_experiments_from_spec",
	"BackendSTTAPI",
]
