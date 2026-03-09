from __future__ import annotations

import unittest

from stt_module.evaluation.metrics import compute_wer_cer


class TestEvaluationMetrics(unittest.TestCase):
    def test_wer_cer_identity(self) -> None:
        out = compute_wer_cer("hello world", "hello world")
        self.assertEqual(out["wer"], 0.0)
        self.assertEqual(out["cer"], 0.0)

    def test_wer_cer_nonzero(self) -> None:
        out = compute_wer_cer("hello world", "goodbye")
        self.assertGreater(out["wer"], 0.0)
        self.assertGreater(out["cer"], 0.0)


if __name__ == "__main__":
    unittest.main()
