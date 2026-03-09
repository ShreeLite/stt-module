from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_stage_latency_breakdown(results_file: str | Path, output_dir: str | Path) -> Path:
    data = _load_json(results_file)
    rows: List[Dict[str, Any]] = data.get("results", [])

    stage_acc: dict[str, list[float]] = {}
    for row in rows:
        for stage, latency in row.get("stage_latencies_ms", {}).items():
            stage_acc.setdefault(stage, []).append(float(latency))

    stages = sorted(stage_acc.keys())
    values = [sum(stage_acc[s]) / len(stage_acc[s]) for s in stages]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart = output_path / "stage_latency_breakdown.png"

    plt.figure(figsize=(10, 5))
    plt.bar(stages, values)
    plt.ylabel("Latency (ms)")
    plt.title("Average Stage Latency")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(chart)
    plt.close()
    return chart


def plot_chunk_boundaries(single_result_file: str | Path, output_dir: str | Path) -> Path:
    data = _load_json(single_result_file)
    waveform = data.get("debug", {}).get("waveform_envelope", [])
    boundaries = data.get("debug", {}).get("chunk_boundaries", [])
    duration = data.get("metrics", {}).get("audio_duration_s", 0.0)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart = output_path / "chunk_boundaries.png"

    plt.figure(figsize=(12, 4))
    if waveform:
        xs = [duration * (i / max(1, len(waveform) - 1)) for i in range(len(waveform))]
        plt.plot(xs, waveform, label="Waveform envelope")
    for seg in boundaries:
        plt.axvspan(seg["start_s"], seg["end_s"], alpha=0.2, color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform and Chunk Boundaries")
    plt.tight_layout()
    plt.savefig(chart)
    plt.close()
    return chart


def plot_config_comparison(results_file: str | Path, output_dir: str | Path) -> Path:
    data = _load_json(results_file)
    rows: List[Dict[str, Any]] = data.get("results", [])

    by_exp: dict[str, dict[str, float]] = {}
    for row in rows:
        exp = row["experiment"]
        ent = by_exp.setdefault(exp, {"lat_sum": 0.0, "conf_sum": 0.0, "count": 0.0})
        ent["lat_sum"] += float(row["latency_ms"])
        ent["conf_sum"] += float(row["confidence"])
        ent["count"] += 1.0

    exps = sorted(by_exp.keys())
    avg_lat = [by_exp[e]["lat_sum"] / by_exp[e]["count"] for e in exps]
    avg_conf = [by_exp[e]["conf_sum"] / by_exp[e]["count"] for e in exps]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart = output_path / "config_comparison.png"

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(exps, avg_lat, color="steelblue", alpha=0.7, label="Avg latency")
    ax2.plot(exps, avg_conf, color="darkgreen", marker="o", label="Avg confidence")
    ax1.set_ylabel("Latency (ms)")
    ax2.set_ylabel("Confidence")
    ax1.set_title("Experiment Configuration Comparison")
    ax1.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    plt.savefig(chart)
    plt.close(fig)
    return chart
