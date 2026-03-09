from __future__ import annotations

import time
from typing import Callable, TypeVar

from stt_module.config import STTConfig
from stt_module.logging_utils import (
    log_chunking_strategy,
    log_pipeline_end,
    log_pipeline_start,
    log_silence_decision,
)
from stt_module.models import (
    AudioData,
    ChunkTranscript,
    DebugInfo,
    PipelineMetrics,
    STTResult,
    StageMetric,
    SpeechSegment,
)
from stt_module.stages.chunking import Chunker
from stt_module.stages.confidence import ConfidenceFilter
from stt_module.stages.noise_reduction import NoiseReducer
from stt_module.stages.postprocess import TranscriptPostProcessor
from stt_module.stages.preprocessing import Preprocessor
from stt_module.stages.recognition import SpeechRecognizer
from stt_module.stages.vad import VoiceActivityDetector
from stt_module.utils.audio import compute_waveform_envelope, rms

T = TypeVar("T")


class STTPipeline:
    def __init__(self, logger) -> None:
        self.logger = logger
        self.preprocessor = Preprocessor()
        self.noise_reducer = NoiseReducer()
        self.vad = VoiceActivityDetector()
        self.chunker = Chunker()
        self.recognizer = SpeechRecognizer()
        self.postprocessor = TranscriptPostProcessor()
        self.conf_filter = ConfidenceFilter()

    def run(self, audio: AudioData, config: STTConfig) -> STTResult:
        pipeline_start = time.perf_counter()
        stage_metrics: list[StageMetric] = []
        stage_latency_map: dict[str, float] = {}

        log_pipeline_start(self.logger, config.to_dict(), config.model_name)

        processed_audio = audio
        if config.enable_preprocessing:
            processed_audio, metric = self._timed_call("preprocessing", lambda: self.preprocessor.run(audio, config))
            stage_metrics.append(metric)
            stage_latency_map[metric.stage_name] = metric.latency_ms
        else:
            stage_metrics.append(StageMetric("preprocessing", 0.0, skipped=True))

        if config.enable_noise_reduction:
            processed_audio, metric = self._timed_call(
                "noise_reduction", lambda: self.noise_reducer.run(processed_audio, config)
            )
            stage_metrics.append(metric)
            stage_latency_map[metric.stage_name] = metric.latency_ms
        else:
            stage_metrics.append(StageMetric("noise_reduction", 0.0, skipped=True))

        vad_segments: list[SpeechSegment] = []
        if config.enable_vad:
            vad_segments, metric = self._timed_call("vad", lambda: self.vad.detect(processed_audio, config))
            stage_metrics.append(metric)
            stage_latency_map[metric.stage_name] = metric.latency_ms
        else:
            stage_metrics.append(StageMetric("vad", 0.0, skipped=True))

        audio_rms = rms(processed_audio.samples)
        no_speech_detected = self._is_no_speech(processed_audio, config, vad_segments, audio_rms)
        log_silence_decision(self.logger, no_speech_detected=no_speech_detected, audio_rms=audio_rms)

        chunking_strategy, strategy_reason = self._resolve_chunking_strategy(processed_audio, config, vad_segments)
        log_chunking_strategy(self.logger, strategy=chunking_strategy, reason=strategy_reason)

        chunks = []
        if chunking_strategy == "none":
            chunks = [self.chunker.full_audio_chunk(processed_audio)]
            stage_metrics.append(StageMetric("chunking", 0.0, skipped=True))
        elif chunking_strategy == "vad":
            chunks, metric = self._timed_call(
                "chunking", lambda: self.chunker.create_vad_chunks(processed_audio, config, vad_segments)
            )
            stage_metrics.append(metric)
            stage_latency_map[metric.stage_name] = metric.latency_ms
            if not chunks:
                chunks, metric = self._timed_call(
                    "chunking", lambda: self.chunker.create_fixed_chunks(processed_audio, config)
                )
                stage_metrics[-1] = metric
                stage_latency_map[metric.stage_name] = metric.latency_ms
                chunking_strategy = "fixed"
        else:
            chunks, metric = self._timed_call(
                "chunking", lambda: self.chunker.create_fixed_chunks(processed_audio, config)
            )
            stage_metrics.append(metric)
            stage_latency_map[metric.stage_name] = metric.latency_ms

        chunk_transcripts: list[ChunkTranscript] = []
        if no_speech_detected:
            stage_metrics.append(StageMetric("recognition", 0.0, skipped=True))
        else:
            stt_start = time.perf_counter()
            for chunk in chunks:
                chunk_transcripts.append(self.recognizer.transcribe_chunk(chunk, config))
            stt_latency = (time.perf_counter() - stt_start) * 1000.0
            stage_metrics.append(StageMetric("recognition", stt_latency))
            stage_latency_map["recognition"] = stt_latency

        if config.enable_confidence_filtering:
            chunk_transcripts, metric = self._timed_call(
                "confidence_filter", lambda: self.conf_filter.run(chunk_transcripts, config.confidence_threshold)
            )
            stage_metrics.append(metric)
            stage_latency_map[metric.stage_name] = metric.latency_ms
        else:
            stage_metrics.append(StageMetric("confidence_filter", 0.0, skipped=True))

        transcript = " ".join(chunk.text for chunk in chunk_transcripts if chunk.text).strip()
        if config.enable_postprocessing:
            transcript, metric = self._timed_call("postprocessing", lambda: self.postprocessor.run(transcript))
            stage_metrics.append(metric)
            stage_latency_map[metric.stage_name] = metric.latency_ms
        else:
            stage_metrics.append(StageMetric("postprocessing", 0.0, skipped=True))

        overall_conf = self._overall_confidence(chunk_transcripts)

        total_latency_ms = (time.perf_counter() - pipeline_start) * 1000.0
        metrics = PipelineMetrics(
            total_latency_ms=total_latency_ms,
            audio_duration_s=processed_audio.duration_s,
            number_of_chunks=len(chunks),
            model_used=config.model_name,
            overall_confidence=overall_conf,
            chunking_strategy_used=chunking_strategy,
            no_speech_detected=no_speech_detected,
            audio_rms=audio_rms,
            stage_latencies_ms=stage_latency_map,
        )

        debug = DebugInfo(
            vad_segments=vad_segments,
            chunk_boundaries=[SpeechSegment(c.start_s, c.end_s) for c in chunks],
            waveform_envelope=compute_waveform_envelope(processed_audio) if config.enable_debug_visualization else [],
        )

        log_pipeline_end(
            self.logger,
            total_latency_ms=total_latency_ms,
            chunk_count=len(chunks),
            confidence=overall_conf,
        )

        return STTResult(
            transcript=transcript,
            confidence=overall_conf,
            metrics=metrics,
            partial_transcripts=chunk_transcripts,
            stage_metrics=stage_metrics,
            debug=debug,
        )

    def _timed_call(self, stage_name: str, fn: Callable[[], T]) -> tuple[T, StageMetric]:
        start = time.perf_counter()
        output = fn()
        latency = (time.perf_counter() - start) * 1000.0
        return output, StageMetric(stage_name=stage_name, latency_ms=latency)

    def _overall_confidence(self, chunks: list[ChunkTranscript]) -> float:
        if not chunks:
            return 0.0
        return sum(c.confidence for c in chunks) / len(chunks)

    def _is_no_speech(
        self,
        audio: AudioData,
        config: STTConfig,
        vad_segments: list[SpeechSegment],
        audio_rms: float,
    ) -> bool:
        if not config.enable_silence_detection:
            return False
        energy_silent = audio_rms < config.silence_rms_threshold
        vad_silent = config.enable_vad and len(vad_segments) == 0
        if config.enable_vad:
            return energy_silent or vad_silent
        return energy_silent

    def _resolve_chunking_strategy(
        self,
        audio: AudioData,
        config: STTConfig,
        vad_segments: list[SpeechSegment],
    ) -> tuple[str, str]:
        if config.chunking_policy == "manual":
            if not config.enable_chunking:
                return "none", "manual-disabled"
            if config.chunking_mode == "vad":
                if config.enable_vad and vad_segments:
                    return "vad", "manual-vad"
                return "fixed", "manual-vad-fallback-fixed"
            return "fixed", "manual-fixed"

        if audio.duration_s < config.auto_chunking_duration_threshold_s:
            return "none", "auto-short-audio"

        if config.enable_vad and vad_segments:
            return "vad", "auto-long-audio-with-vad"
        return "fixed", "auto-long-audio-fixed-fallback"
