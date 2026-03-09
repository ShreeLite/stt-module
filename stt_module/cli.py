from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from stt_module.compare import compare_configs
from stt_module.service import STTService


def _load_overrides(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone STT experimentation module")
    parser.add_argument("audio", help="Path to WAV audio input")
    parser.add_argument("--config", help="JSON file with config overrides")
    parser.add_argument("--compare-config", help="Second JSON config for comparison mode")

    args = parser.parse_args()

    config_a = _load_overrides(args.config)

    if args.compare_config:
        config_b = _load_overrides(args.compare_config)
        output = compare_configs(args.audio, config_a, config_b)
    else:
        service = STTService()
        output = service.transcribe(args.audio, config_a)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
