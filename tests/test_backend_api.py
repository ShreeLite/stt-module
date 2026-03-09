from __future__ import annotations

import unittest
from unittest.mock import patch

from stt_module.integration.backend_api import BackendSTTAPI


class TestBackendAPI(unittest.TestCase):
    def test_transcribe_ok_status(self) -> None:
        api = BackendSTTAPI(low_confidence_threshold=0.5)
        fake = {
            "transcript": "hello",
            "confidence": 0.8,
            "metrics": {
                "total_latency_ms": 123.0,
                "audio_duration_s": 2.0,
                "number_of_chunks": 1,
                "chunking_strategy_used": "none",
                "no_speech_detected": False,
                "stage_latencies_ms": {"recognition": 100.0},
            },
            "partial_transcripts": [],
            "debug": {},
        }

        with patch.object(api.service, "transcribe", return_value=fake):
            out = api.transcribe("audio.wav")

        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["warnings"], [])

    def test_transcribe_no_speech_status(self) -> None:
        api = BackendSTTAPI(low_confidence_threshold=0.5)
        fake = {
            "transcript": "",
            "confidence": 0.0,
            "metrics": {
                "total_latency_ms": 15.0,
                "audio_duration_s": 2.0,
                "number_of_chunks": 1,
                "chunking_strategy_used": "none",
                "no_speech_detected": True,
                "stage_latencies_ms": {},
            },
            "partial_transcripts": [],
            "debug": {},
        }

        with patch.object(api.service, "transcribe", return_value=fake):
            out = api.transcribe("audio.wav")

        self.assertEqual(out["status"], "no_speech")
        self.assertTrue(any("No meaningful speech" in w for w in out["warnings"]))


if __name__ == "__main__":
    unittest.main()
