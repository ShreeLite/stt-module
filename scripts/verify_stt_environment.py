from __future__ import annotations

import argparse
import json
from pathlib import Path

from stt_module.config import STTConfig
from stt_module.stages.recognition import SpeechRecognizer
from stt_module.utils.audio import read_audio_file


def verify_environment(audio_file: Path, model_name: str, device: str) -> dict:
    checks: dict[str, str | bool | dict] = {
        "import_ok": True,
        "audio_decode_ok": False,
        "model_init_ok": False,
    }

    audio = read_audio_file(audio_file)
    checks["audio_decode_ok"] = audio.samples.size > 0 and audio.sample_rate_hz > 0
    checks["audio_info"] = {
        "sample_rate_hz": audio.sample_rate_hz,
        "duration_s": round(audio.duration_s, 3),
        "channels": audio.channels,
    }

    recognizer = SpeechRecognizer()
    cfg = STTConfig(model_name=model_name, model_device=device)
    recognizer._get_model(cfg)
    checks["model_init_ok"] = True
    checks["model"] = {"name": model_name, "device": device}

    checks["all_ok"] = bool(checks["import_ok"] and checks["audio_decode_ok"] and checks["model_init_ok"])
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify STT module runtime environment")
    parser.add_argument("--audio", required=True, help="Path to a sample audio file")
    parser.add_argument("--model", default="tiny", help="Whisper model name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Model device")
    args = parser.parse_args()

    result = verify_environment(Path(args.audio), args.model, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
