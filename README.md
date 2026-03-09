# Standalone Modular STT Module

A self-contained speech-to-text experimentation system with configurable, independently toggleable pipeline stages.

This implementation is production-oriented and uses real components:
- audio decoding and resampling via `soundfile` + `scipy`
- optional spectral noise reduction via `noisereduce`
- voice activity detection via `webrtcvad`
- speech recognition via `faster-whisper`

It also supports experimentation workflows:
- automatic chunking strategy selection for short vs long audio
- silence/no-speech detection before inference
- batch experiment runs with JSON/YAML specs
- WER/CER evaluation via `jiwer`
- visualization utilities for latency, chunk boundaries, and config comparison

## Features

- Modular pipeline stages:
  - preprocessing
  - optional noise reduction
  - optional voice activity detection (VAD)
  - optional chunking (VAD-based preferred, fixed-duration fallback)
  - speech recognition engine
  - optional transcript post-processing
  - optional confidence filtering
- Runtime configuration changes per request (no restart required)
- Manual or automatic chunking policy (`manual` / `auto`)
- Streaming-style chunk transcription with partial outputs
- Stage-level and total latency metrics
- Comparison mode for two configs on the same audio
- Logging of enabled modules, configuration, model, latency, and chunk counts
- Frontend payload adapter for UI integration

## Install

```bash
pip install -e .
```

Migration-friendly imports:

```python
from stt_module import STTService
# or compatibility alias
from stt import STTService
```

## Usage

```bash
stt-module transcribe path/to/audio.wav
```

With config overrides:

```bash
stt-module transcribe path/to/audio.wav --config config_a.json
```

Comparison mode:

```bash
stt-module compare path/to/audio.wav --config config_a.json --compare-config config_b.json
```

Batch experiments from spec:

```bash
stt-module experiments --spec experiments.yaml
```

WER/CER evaluation:

```bash
stt-module evaluate --reference "hello world" --hypothesis "hello ward"
```

Generate plots:

```bash
stt-module visualize --input experiment_results.json --output-dir plots --kind all
```

Dataset-level evaluation:

```bash
stt-module evaluate-dataset --audio-dir ./dataset --mapping ./mapping.json --output dataset_eval.json
```

Environment verification:

```bash
python scripts/verify_stt_environment.py --audio voice-sample.wav --model tiny --device cpu
```

## Example Config

```json
{
  "enable_noise_reduction": true,
  "enable_vad": true,
  "chunking_policy": "auto",
  "vad_sensitivity": 0.62,
  "model_name": "small",
  "enable_debug_visualization": true,
  "confidence_threshold": 0.5
}
```

## Output Schema (summary)

- `transcript`
- `confidence`
- `metrics`
  - `total_latency_ms`
  - `audio_duration_s`
  - `number_of_chunks`
  - `model_used`
  - `overall_confidence`
  - `chunking_strategy_used`
  - `no_speech_detected`
  - `audio_rms`
  - `stage_latencies_ms`
- `partial_transcripts`
- `stage_metrics`
- `debug`
  - `vad_segments`
  - `chunk_boundaries`
  - `waveform_envelope`

## Notes

- The recognizer uses Faster-Whisper models (default: `small`).
- Use VAD-based chunking when possible to preserve speech boundaries and reduce context loss.
- Fixed chunking supports overlap to preserve context across chunk boundaries.

## Automatic Chunking Behavior

- `chunking_policy="manual"`: honors `enable_chunking` and `chunking_mode`.
- `chunking_policy="auto"`:
  - short audio (`duration < auto_chunking_duration_threshold_s`) => no chunking
  - long audio with VAD speech segments => VAD chunking
  - long audio without usable VAD => fixed chunking fallback

The selected strategy is stored in `metrics.chunking_strategy_used`.

## Silence Detection

- If `enable_silence_detection=true`, the pipeline evaluates low-energy/no-speech conditions.
- When no meaningful speech is detected, recognition is skipped and output includes:
  - `metrics.no_speech_detected=true`
  - `metrics.audio_rms=<value>`

## Experiment Spec (JSON/YAML)

```yaml
audio_input: voice-sample.wav
output_file: experiment_results.json
print_table: true
ground_truths:
  voice-sample.wav: "Hi there, this is a sample voice recording ..."
experiments:
  - name: baseline
    overrides:
      model_name: tiny
      enable_vad: false
      enable_chunking: false
  - name: auto_chunking
    overrides:
      model_name: tiny
      chunking_policy: auto
```

## Visualization Outputs

- `stage_latency_breakdown.png`
- `chunk_boundaries.png`
- `config_comparison.png`

## Backend Facade and Endpoint References

- Stable backend facade: `stt_module/integration/backend_api.py`
- Generic endpoint examples: `stt_module/integration/backend_examples.py`

## CI Testing Profiles

- Fast profile: `make test-fast`
- Full profile: `make test-full`

Fast is intended for frequent commit checks; full includes heavier integration coverage.

## Handoff Documentation

- Architecture: `docs/STT_ARCHITECTURE.md`
- Integration/API contract: `docs/STT_INTEGRATION_GUIDE.md`

## Project Structure

- `stt_module/pipeline.py`: orchestration and robustness logic
- `stt_module/experiments/runner.py`: batch experimentation engine
- `stt_module/evaluation/metrics.py`: WER/CER metrics
- `stt_module/visualization/plots.py`: charts and diagnostics
- `stt_module/integration/frontend.py`: frontend-friendly payload adapter

## Production Guidance

- Use `model_device="cuda"` and `model_compute_type="float16"` on compatible GPUs.
- Keep `condition_on_previous_text=false` to avoid context leakage between chunks unless explicitly desired.
- Avoid very small chunks; use VAD chunking or larger fixed windows with overlap.
