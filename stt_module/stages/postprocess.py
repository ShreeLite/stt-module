from __future__ import annotations

import re


class TranscriptPostProcessor:
    name = "postprocessing"

    def run(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("[silence]", "")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        normalized = text[0].upper() + text[1:]
        if normalized[-1] not in ".!?":
            normalized += "."
        return normalized
