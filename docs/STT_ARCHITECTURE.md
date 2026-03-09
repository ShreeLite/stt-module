# STT Module Architecture

## Overview

`stt_module` is a modular internal library for speech-to-text experimentation and production inference workflows. It is designed to be imported directly into an existing backend repository without requiring a separate microservice.

## Core Pipeline

Pipeline order:
1. preprocessing
2. optional noise reduction
3. optional VAD
4. chunking strategy resolution (manual/auto)
5. chunk creation (none/vad/fixed)
6. optional silence short-circuit (skip recognition)
7. recognition (faster-whisper)
8. optional confidence filtering
9. optional post-processing

### Chunking Strategies

- `manual`: honors explicit config toggles.
- `auto`: disables chunking for short audio and enables chunking for long audio.
- strategy selection records: `none`, `vad`, or `fixed` in metrics.

### Silence Detection

- Uses RMS threshold and optional VAD evidence.
- If no speech is detected, recognition is skipped.
- Output includes `no_speech_detected` and `audio_rms`.

## Configuration System

Main configuration object: `stt_module.config.STTConfig`

Key controls:
- model (`model_name`, `model_device`, `model_compute_type`)
- chunking (`chunking_policy`, `chunking_mode`, overlap/duration controls)
- VAD sensitivity and frame sizing
- confidence threshold
- silence detection thresholds

## Metrics and Observability

Output includes:
- transcript and confidence
- total latency and stage latencies
- audio duration and chunk count
- chunking strategy used
- no-speech decision and audio RMS
- debug payload (vad segments, chunk boundaries, waveform envelope)

## Experimentation and Evaluation

- Experiment runner: `stt_module.experiments.runner`
  - batch runs over file or directory
  - JSON/YAML experiment specs
  - tabular summaries and JSON artifacts
- Evaluation:
  - pairwise WER/CER via `jiwer`
  - dataset-level evaluation with aggregate metrics

## Visualization

`stt_module.visualization.plots` generates:
- stage latency breakdown charts
- waveform + chunk boundary charts
- config comparison charts

## Integration Layer

- `stt_module.integration.backend_api.BackendSTTAPI`
  - stable response contract for backend routes
  - warnings/status normalization
- `stt_module.integration.backend_examples`
  - generic route-handler reference patterns
