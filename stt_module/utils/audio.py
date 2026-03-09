from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from stt_module.models import AudioData


def read_audio_file(path: str | Path) -> AudioData:
    samples, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    channels = samples.shape[1]
    mono_samples = samples.mean(axis=1) if channels > 1 else samples[:, 0]

    return AudioData(
        samples=mono_samples,
        sample_rate_hz=int(sample_rate),
        channels=1,
        sample_width_bytes=2,
    )


def read_audio_input(audio_input: str | Path | AudioData) -> AudioData:
    if isinstance(audio_input, AudioData):
        if not isinstance(audio_input.samples, np.ndarray):
            return AudioData(
                samples=np.asarray(audio_input.samples, dtype=np.float32),
                sample_rate_hz=audio_input.sample_rate_hz,
                channels=audio_input.channels,
                sample_width_bytes=audio_input.sample_width_bytes,
            )
        return audio_input
    return read_audio_file(audio_input)


def ensure_format(audio: AudioData, target_sample_rate_hz: int, target_channels: int) -> AudioData:
    samples = np.asarray(audio.samples, dtype=np.float32)

    if audio.channels > 1 and target_channels == 1:
        samples = samples.mean(axis=1)

    if samples.ndim > 1:
        samples = samples[:, 0]

    max_abs = float(np.max(np.abs(samples))) if samples.size else 0.0
    if max_abs > 1.0:
        samples = samples / max_abs

    sample_rate = audio.sample_rate_hz
    if sample_rate != target_sample_rate_hz:
        samples = _resample(samples, sample_rate, target_sample_rate_hz)
        sample_rate = target_sample_rate_hz

    return AudioData(
        samples=samples,
        sample_rate_hz=sample_rate,
        channels=target_channels,
        sample_width_bytes=2,
    )


def compute_waveform_envelope(audio: AudioData, buckets: int = 80) -> list[float]:
    if audio.samples.size == 0:
        return []
    bucket_size = max(1, audio.samples.shape[0] // buckets)
    envelope: list[float] = []
    for idx in range(0, audio.samples.shape[0], bucket_size):
        block = audio.samples[idx : idx + bucket_size]
        if block.size == 0:
            continue
        peak = float(np.max(np.abs(block)))
        envelope.append(round(min(1.0, peak), 5))
    return envelope


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float32), dtype=np.float32)))


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples

    # Polyphase resampling for high-quality sample-rate conversion.
    gcd = np.gcd(src_rate, dst_rate)
    up = dst_rate // gcd
    down = src_rate // gcd
    return resample_poly(samples, up, down).astype(np.float32)
