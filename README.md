# Standalone Modular STT Module

A self-contained speech-to-text experimentation system with configurable, independently toggleable pipeline stages.

This implementation is production-oriented and uses real components:
- audio decoding and resampling via `soundfile` + `scipy`
- optional spectral noise reduction via `noisereduce`
- voice activity detection via `webrtcvad`
- speech recognition via `faster-whisper`

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
- Streaming-style chunk transcription with partial outputs
- Stage-level and total latency metrics
- Comparison mode for two configs on the same audio
- Logging of enabled modules, configuration, model, latency, and chunk counts

## Install

```bash
pip install -e .
```

## Usage

```bash
stt-module path/to/audio.wav
```

With config overrides:

```bash
stt-module path/to/audio.wav --config config_a.json
```

Comparison mode:

```bash
stt-module path/to/audio.wav --config config_a.json --compare-config config_b.json
```

## Example Config

```json
{
  "enable_noise_reduction": true,
  "enable_vad": true,
  "enable_chunking": true,
  "chunking_mode": "vad",
  "vad_sensitivity": 0.62,
  "model_name": "simple",
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

## Production Guidance

- Use `model_device="cuda"` and `model_compute_type="float16"` on compatible GPUs.
- Keep `condition_on_previous_text=false` to avoid context leakage between chunks unless explicitly desired.
- Avoid very small chunks; use VAD chunking or larger fixed windows with overlap.
