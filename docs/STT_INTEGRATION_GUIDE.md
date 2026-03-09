# STT Integration Guide

## Backend Integration (Internal Library)

Install dependencies and import directly:

```python
from stt_module import STTService
from stt_module.integration.backend_api import BackendSTTAPI
```

Use `BackendSTTAPI` in route handlers for stable contracts.

## Environment Verification

Run:

```bash
python scripts/verify_stt_environment.py --audio voice-sample.wav --model tiny --device cpu
```

Checks:
- module import
- audio decode
- model initialization

## Backend API Contract

### Transcribe Request

Request body (example):

```json
{
  "audioPath": "/uploads/user123.wav",
  "config": {
    "model_name": "tiny",
    "chunking_policy": "auto"
  }
}
```

Response body (example):

```json
{
  "status": "ok",
  "transcript": "...",
  "confidence": 0.78,
  "latency_ms": 2100.5,
  "audio_duration_s": 26.3,
  "chunk_count": 1,
  "chunking_strategy": "none",
  "stage_latencies_ms": {
    "preprocessing": 30.1,
    "recognition": 2050.3
  },
  "warnings": []
}
```

Possible `status` values:
- `ok`
- `no_speech`
- `low_confidence`

### Comparison Request

Request body (example):

```json
{
  "audioPath": "/uploads/user123.wav",
  "configA": {"enable_vad": true, "enable_chunking": true, "chunking_mode": "vad"},
  "configB": {"enable_vad": false, "enable_chunking": false}
}
```

Response body (example):

```json
{
  "status": "ok",
  "mode": "comparison",
  "pipelineA": {"status": "ok", "transcript": "...", "confidence": 0.76},
  "pipelineB": {"status": "ok", "transcript": "...", "confidence": 0.77}
}
```

## Frontend Integration Guidance

Frontend should:
- upload audio file to backend endpoint
- display transcript and confidence
- show latency and chunk count
- visualize stage latency breakdown as bars
- show warnings for `no_speech` and `low_confidence`

Comparison mode UI should display side-by-side cards:
- transcript A vs B
- confidence A vs B
- latency A vs B
- chunk count and strategy A vs B

## Dataset Evaluation Workflow

Inputs:
- audio directory
- transcript mapping file (JSON/YAML/CSV)

CLI:

```bash
stt-module evaluate-dataset --audio-dir ./dataset --mapping ./mapping.json --output dataset_eval.json
```

Outputs:
- per-file WER/CER
- per-file confidence/latency
- aggregate WER/CER/latency/confidence
- total evaluation runtime

## Test Profiles

Fast profile (commit-level):

```bash
make test-fast
```

Full profile (periodic / release-level):

```bash
make test-full
```

## Reference Backend Handlers

See: `stt_module/integration/backend_examples.py`
