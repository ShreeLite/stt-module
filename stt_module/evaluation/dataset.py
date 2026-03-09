from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from stt_module.evaluation.metrics import compute_wer_cer
from stt_module.service import STTService


def evaluate_dataset(
    audio_dir: str | Path,
    transcript_map_file: str | Path,
    config_overrides: Dict[str, Any] | None = None,
    output_file: str | Path | None = None,
) -> Dict[str, Any]:
    service = STTService()
    mapping = _load_mapping(transcript_map_file)
    base = Path(audio_dir)

    start = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for name, reference in mapping.items():
        audio_path = base / name
        if not audio_path.exists():
            rows.append(
                {
                    "audio_file": str(audio_path),
                    "status": "missing_audio",
                    "reference": reference,
                }
            )
            continue

        result = service.transcribe(audio_path, config_overrides)
        quality = compute_wer_cer(reference=reference, hypothesis=result["transcript"])

        rows.append(
            {
                "audio_file": str(audio_path),
                "status": "ok",
                "reference": reference,
                "hypothesis": result["transcript"],
                "wer": quality["wer"],
                "cer": quality["cer"],
                "confidence": result["confidence"],
                "latency_ms": result["metrics"]["total_latency_ms"],
                "chunk_count": result["metrics"]["number_of_chunks"],
                "chunking_strategy": result["metrics"].get("chunking_strategy_used", "none"),
            }
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    scored = [r for r in rows if r.get("status") == "ok"]

    summary: Dict[str, Any] = {
        "dataset_size": len(mapping),
        "evaluated_count": len(scored),
        "total_evaluation_time_ms": round(elapsed_ms, 3),
    }
    if scored:
        summary.update(
            {
                "avg_wer": round(sum(r["wer"] for r in scored) / len(scored), 4),
                "avg_cer": round(sum(r["cer"] for r in scored) / len(scored), 4),
                "avg_latency_ms": round(sum(r["latency_ms"] for r in scored) / len(scored), 3),
                "avg_confidence": round(sum(r["confidence"] for r in scored) / len(scored), 4),
            }
        )

    out = {"results": rows, "summary": summary}
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _load_mapping(path: str | Path) -> Dict[str, str]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}

    if p.suffix.lower() == ".csv":
        mapping: Dict[str, str] = {}
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[str(row["audio_file"])]=str(row["transcript"])
        return mapping

    raise ValueError("Unsupported transcript mapping format. Use JSON, YAML, or CSV.")
