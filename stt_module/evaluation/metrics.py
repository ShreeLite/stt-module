from __future__ import annotations

from typing import Dict

from jiwer import cer, wer


def compute_wer_cer(reference: str, hypothesis: str) -> Dict[str, float]:
    return {
        "wer": float(wer(reference, hypothesis)),
        "cer": float(cer(reference, hypothesis)),
    }
