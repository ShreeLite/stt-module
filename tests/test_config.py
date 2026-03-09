from __future__ import annotations

import unittest

from stt_module.config import STTConfig


class TestConfig(unittest.TestCase):
    def test_invalid_overlap_raises(self) -> None:
        with self.assertRaises(ValueError):
            STTConfig(fixed_chunk_duration_s=2.0, fixed_chunk_overlap_s=2.0)

    def test_invalid_confidence_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            STTConfig(confidence_threshold=1.1)

    def test_merge_updates_fields(self) -> None:
        cfg = STTConfig()
        merged = cfg.merged({"model_name": "base", "enable_vad": False})
        self.assertEqual(merged.model_name, "base")
        self.assertFalse(merged.enable_vad)


if __name__ == "__main__":
    unittest.main()
