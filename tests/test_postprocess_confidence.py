from __future__ import annotations

import unittest

from stt_module.models import ChunkTranscript
from stt_module.stages.confidence import ConfidenceFilter
from stt_module.stages.postprocess import TranscriptPostProcessor


class TestPostprocessAndConfidence(unittest.TestCase):
    def test_postprocessing_normalizes_and_punctuates(self) -> None:
        text = "   hello   world  "
        processed = TranscriptPostProcessor().run(text)
        self.assertEqual(processed, "Hello world.")

    def test_postprocessing_removes_silence_token(self) -> None:
        text = "[silence] hello"
        processed = TranscriptPostProcessor().run(text)
        self.assertEqual(processed, "Hello.")

    def test_confidence_filter_applies_threshold(self) -> None:
        items = [
            ChunkTranscript(0, "low", 0.2, 0.0, 1.0),
            ChunkTranscript(1, "high", 0.8, 1.0, 2.0),
        ]
        filtered = ConfidenceFilter().run(items, threshold=0.5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].text, "high")


if __name__ == "__main__":
    unittest.main()
