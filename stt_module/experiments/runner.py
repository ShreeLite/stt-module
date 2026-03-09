from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from tabulate import tabulate

from stt_module.evaluation.metrics import compute_wer_cer
from stt_module.service import STTService


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    overrides: Dict[str, Any]


class ExperimentRunner:
    def __init__(self, service: STTService | None = None) -> None:
        self.service = service or STTService()

    def run(
        self,
        audio_input: str | Path,
        experiments: List[ExperimentConfig],
        ground_truths: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        audio_files = self._resolve_audio_files(audio_input)
        gt_map = ground_truths or {}

        rows: list[dict[str, Any]] = []
        for audio_file in audio_files:
            audio_key = audio_file.name
            reference = gt_map.get(audio_key)

            for exp in experiments:
                output = self.service.transcribe(audio_file, exp.overrides)
                row: dict[str, Any] = {
                    "audio_file": str(audio_file),
                    "audio_name": audio_key,
                    "experiment": exp.name,
                    "config": exp.overrides,
                    "transcript": output["transcript"],
                    "confidence": output["confidence"],
                    "latency_ms": output["metrics"]["total_latency_ms"],
                    "chunk_count": output["metrics"]["number_of_chunks"],
                    "chunking_strategy": output["metrics"].get("chunking_strategy_used", "none"),
                    "no_speech_detected": output["metrics"].get("no_speech_detected", False),
                    "stage_latencies_ms": output["metrics"].get("stage_latencies_ms", {}),
                    "model_used": output["metrics"].get("model_used"),
                }

                if reference is not None:
                    quality = compute_wer_cer(reference=reference, hypothesis=output["transcript"])
                    row.update(quality)
                rows.append(row)

        return {"results": rows, "summary": self._summary(rows)}

    def write_results(self, result: Dict[str, Any], output_file: str | Path) -> Path:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return output_path

    def render_table(self, result: Dict[str, Any]) -> str:
        rows = result.get("results", [])
        flat = [
            {
                "audio": r["audio_name"],
                "exp": r["experiment"],
                "model": r.get("model_used"),
                "lat_ms": r["latency_ms"],
                "conf": r["confidence"],
                "chunks": r["chunk_count"],
                "chunking": r.get("chunking_strategy"),
                "wer": r.get("wer", "-"),
                "cer": r.get("cer", "-"),
            }
            for r in rows
        ]
        if not flat:
            return "No results"
        return tabulate(flat, headers="keys", tablefmt="github", floatfmt=".3f")

    def _resolve_audio_files(self, audio_input: str | Path) -> list[Path]:
        path = Path(audio_input)
        if path.is_file():
            return [path]
        supported = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
        files = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in supported and p.is_file()]
        return files

    def _summary(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        row_list = list(rows)
        if not row_list:
            return {"count": 0}

        avg_latency = sum(r["latency_ms"] for r in row_list) / len(row_list)
        avg_conf = sum(r["confidence"] for r in row_list) / len(row_list)
        with_wer = [r["wer"] for r in row_list if "wer" in r]
        with_cer = [r["cer"] for r in row_list if "cer" in r]

        summary: Dict[str, Any] = {
            "count": len(row_list),
            "avg_latency_ms": round(avg_latency, 3),
            "avg_confidence": round(avg_conf, 4),
        }
        if with_wer:
            summary["avg_wer"] = round(sum(with_wer) / len(with_wer), 4)
        if with_cer:
            summary["avg_cer"] = round(sum(with_cer) / len(with_cer), 4)
        return summary


def load_experiment_spec(spec_file: str | Path) -> Dict[str, Any]:
    spec_path = Path(spec_file)
    text = spec_path.read_text(encoding="utf-8")
    if spec_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Experiment spec must be a JSON/YAML object")
    return data


def run_experiments_from_spec(spec_file: str | Path) -> Dict[str, Any]:
    spec = load_experiment_spec(spec_file)

    audio_input = spec["audio_input"]
    experiments_raw = spec.get("experiments", [])
    experiments = [ExperimentConfig(name=e["name"], overrides=e.get("overrides", {})) for e in experiments_raw]

    ground_truths = spec.get("ground_truths")
    runner = ExperimentRunner()
    results = runner.run(audio_input=audio_input, experiments=experiments, ground_truths=ground_truths)

    output_file = spec.get("output_file")
    if output_file:
        runner.write_results(results, output_file)

    if spec.get("print_table", True):
        print(runner.render_table(results))

    return results
