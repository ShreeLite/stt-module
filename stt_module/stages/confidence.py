from __future__ import annotations

from stt_module.models import ChunkTranscript


class ConfidenceFilter:
    name = "confidence_filter"

    def run(self, transcripts: list[ChunkTranscript], threshold: float) -> list[ChunkTranscript]:
        filtered: list[ChunkTranscript] = []
        for chunk in transcripts:
            if chunk.confidence >= threshold:
                filtered.append(chunk)
        return filtered
