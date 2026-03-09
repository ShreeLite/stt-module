#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./.venv/bin/python -m unittest \
  tests.test_config \
  tests.test_chunking \
  tests.test_pipeline \
  tests.test_pipeline_robustness \
  tests.test_preprocessing \
  tests.test_postprocess_confidence \
  tests.test_compare \
  tests.test_evaluation_metrics \
  tests.test_experiment_runner \
  tests.test_frontend_integration -v
