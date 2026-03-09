from __future__ import annotations

import math

import numpy as np
from faster_whisper import WhisperModel

from stt_module.config import STTConfig
from stt_module.models import Chunk, ChunkTranscript


class SpeechRecognizer:
    name = "recognition"

    def __init__(self) -> None:
        self._loaded_model_key: tuple[str, str, str] | None = None
        self._model: WhisperModel | None = None

    def _get_model(self, config: STTConfig) -> WhisperModel:
        model_key = (config.model_name, config.model_device, config.model_compute_type)
        if self._model is not None and self._loaded_model_key == model_key:
            return self._model

        self._model = WhisperModel(
            config.model_name,
            device=config.model_device,
            compute_type=config.model_compute_type,
        )
        self._loaded_model_key = model_key
        return self._model

    def transcribe_chunk(self, chunk: Chunk, config: STTConfig) -> ChunkTranscript:
        model = self._get_model(config)
        audio = np.asarray(chunk.samples, dtype=np.float32)
        if audio.size == 0:
            return ChunkTranscript(
                chunk_index=chunk.index,
                text="",
                confidence=0.0,
                start_s=chunk.start_s,
                end_s=chunk.end_s,
            )

        segments, _ = model.transcribe(
            audio,
            language=config.language,
            task=config.task,
            beam_size=config.model_beam_size,
            condition_on_previous_text=config.condition_on_previous_text,
            vad_filter=False,
        )

        seg_list = list(segments)
        text = " ".join(seg.text.strip() for seg in seg_list if seg.text and seg.text.strip()).strip()

        if seg_list:
            probs = [math.exp(seg.avg_logprob) for seg in seg_list if hasattr(seg, "avg_logprob")]
            confidence = float(sum(probs) / len(probs)) if probs else 0.0
        else:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return ChunkTranscript(
            chunk_index=chunk.index,
            text=text,
            confidence=confidence,
            start_s=chunk.start_s,
            end_s=chunk.end_s,
        )
