from __future__ import annotations

import unittest

from stt_module.integration.frontend import to_frontend_payload


class TestFrontendIntegration(unittest.TestCase):
    def test_frontend_payload_ok(self) -> None:
        payload = to_frontend_payload(
            {
                "transcript": "hello",
                "confidence": 0.7,
                "metrics": {
                    "total_latency_ms": 120,
                    "audio_duration_s": 4.0,
                    "number_of_chunks": 1,
                    "chunking_strategy_used": "none",
                    "no_speech_detected": False,
                },
                "partial_transcripts": [],
                "debug": {},
            }
        )
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["performance"]["chunkingStrategy"], "none")

    def test_frontend_payload_no_speech(self) -> None:
        payload = to_frontend_payload(
            {
                "transcript": "",
                "confidence": 0.0,
                "metrics": {
                    "total_latency_ms": 20,
                    "audio_duration_s": 4.0,
                    "number_of_chunks": 1,
                    "chunking_strategy_used": "none",
                    "no_speech_detected": True,
                },
                "partial_transcripts": [],
                "debug": {},
            }
        )
        self.assertEqual(payload["status"], "no_speech")


if __name__ == "__main__":
    unittest.main()
