from __future__ import annotations

import array

import numpy as np
import webrtcvad

from stt_module.config import STTConfig
from stt_module.models import AudioData, SpeechSegment


class VoiceActivityDetector:
    name = "vad"

    def detect(self, audio: AudioData, config: STTConfig) -> list[SpeechSegment]:
        if config.vad_frame_ms not in (10, 20, 30):
            raise ValueError("vad_frame_ms must be one of 10, 20, or 30 for webrtcvad")
        if audio.sample_rate_hz not in (8000, 16000, 32000, 48000):
            raise ValueError("audio.sample_rate_hz must be 8k,16k,32k,48k for webrtcvad")

        aggressiveness = int(round((1.0 - config.vad_sensitivity) * 3.0))
        aggressiveness = max(0, min(3, aggressiveness))
        vad = webrtcvad.Vad(aggressiveness)

        frame_samples = max(1, int(audio.sample_rate_hz * (config.vad_frame_ms / 1000.0)))
        min_speech_frames = max(1, config.vad_min_speech_ms // config.vad_frame_ms)
        min_silence_frames = max(1, config.vad_min_silence_ms // config.vad_frame_ms)

        # webrtcvad consumes PCM int16 bytes.
        pcm = np.clip(audio.samples, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)

        speech_flags: list[bool] = []
        for i in range(0, len(audio.samples), frame_samples):
            frame = pcm_i16[i : i + frame_samples]
            if frame.shape[0] != frame_samples:
                break
            speech_flags.append(vad.is_speech(array.array("h", frame.tolist()).tobytes(), audio.sample_rate_hz))

        segments: list[SpeechSegment] = []
        in_speech = False
        speech_start = 0
        speech_frames = 0
        silence_run = 0

        for idx, is_speech in enumerate(speech_flags):
            if is_speech:
                if not in_speech:
                    in_speech = True
                    speech_start = idx
                    speech_frames = 0
                speech_frames += 1
                silence_run = 0
                continue

            if in_speech:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    if speech_frames >= min_speech_frames:
                        start_s = (speech_start * frame_samples) / audio.sample_rate_hz
                        end_s = (idx * frame_samples) / audio.sample_rate_hz
                        segments.append(SpeechSegment(start_s=start_s, end_s=end_s))
                    in_speech = False
                    speech_frames = 0
                    silence_run = 0

        if in_speech and speech_frames >= min_speech_frames:
            start_s = (speech_start * frame_samples) / audio.sample_rate_hz
            end_s = len(audio.samples) / audio.sample_rate_hz
            segments.append(SpeechSegment(start_s=start_s, end_s=end_s))

        return segments
