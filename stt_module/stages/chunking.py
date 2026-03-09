from __future__ import annotations

from stt_module.config import STTConfig
from stt_module.models import AudioData, Chunk, SpeechSegment


class Chunker:
    name = "chunking"

    def full_audio_chunk(self, audio: AudioData) -> Chunk:
        return self._make_chunk(0, 0.0, audio.duration_s, audio)

    def create_chunks(
        self,
        audio: AudioData,
        config: STTConfig,
        vad_segments: list[SpeechSegment] | None,
    ) -> list[Chunk]:
        if audio.duration_s <= config.short_audio_no_chunk_threshold_s:
            return [self._make_chunk(0, 0.0, audio.duration_s, audio)]

        if config.chunking_mode == "vad":
            if not vad_segments:
                return []
            return self._vad_chunks(audio, config, vad_segments)

        return self._fixed_chunks(audio, config)

    def _vad_chunks(
        self,
        audio: AudioData,
        config: STTConfig,
        segments: list[SpeechSegment],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        current_start = None
        current_end = None
        chunk_index = 0

        for seg in segments:
            if current_start is None:
                current_start = seg.start_s
                current_end = seg.end_s
                continue

            gap = seg.start_s - current_end
            merged_duration = seg.end_s - current_start
            if gap <= config.vad_group_pause_s and merged_duration <= config.vad_max_chunk_s:
                current_end = seg.end_s
                continue

            chunks.append(self._make_chunk(chunk_index, current_start, current_end, audio))
            chunk_index += 1
            current_start = seg.start_s
            current_end = seg.end_s

        if current_start is not None and current_end is not None:
            chunks.append(self._make_chunk(chunk_index, current_start, current_end, audio))

        return [c for c in chunks if c.duration_s >= config.min_chunk_duration_s]

    def _fixed_chunks(self, audio: AudioData, config: STTConfig) -> list[Chunk]:
        chunk_dur = max(config.min_chunk_duration_s, config.fixed_chunk_duration_s)
        overlap = max(0.0, min(config.fixed_chunk_overlap_s, chunk_dur - 0.1))
        stride = max(0.1, chunk_dur - overlap)

        chunks: list[Chunk] = []
        start = 0.0
        idx = 0
        while start < audio.duration_s:
            end = min(audio.duration_s, start + chunk_dur)
            chunks.append(self._make_chunk(idx, start, end, audio))
            if end >= audio.duration_s:
                break
            start += stride
            idx += 1
        return chunks

    def _make_chunk(self, index: int, start_s: float, end_s: float, audio: AudioData) -> Chunk:
        start_i = int(start_s * audio.sample_rate_hz)
        end_i = int(end_s * audio.sample_rate_hz)
        return Chunk(
            index=index,
            start_s=start_s,
            end_s=end_s,
            samples=audio.samples[start_i:end_i],
        )
