from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from stt_module.compare import compare_configs
from stt_module.evaluation.dataset import evaluate_dataset
from stt_module.evaluation.metrics import compute_wer_cer
from stt_module.experiments.runner import run_experiments_from_spec
from stt_module.service import STTService
from stt_module.visualization.plots import (
    plot_chunk_boundaries,
    plot_config_comparison,
    plot_stage_latency_breakdown,
)


def _load_overrides(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone STT experimentation module")
    sub = parser.add_subparsers(dest="command")

    transcribe = sub.add_parser("transcribe", help="Run single-file transcription")
    transcribe.add_argument("audio", help="Path to audio input")
    transcribe.add_argument("--config", help="JSON file with config overrides")

    compare = sub.add_parser("compare", help="Compare two pipeline configurations")
    compare.add_argument("audio", help="Path to audio input")
    compare.add_argument("--config", required=True, help="JSON file for pipeline A")
    compare.add_argument("--compare-config", required=True, help="JSON file for pipeline B")

    experiments = sub.add_parser("experiments", help="Run batch experiments from JSON/YAML spec")
    experiments.add_argument("--spec", required=True, help="Path to experiment spec file")

    evaluate = sub.add_parser("evaluate", help="Compute WER/CER from reference and hypothesis text")
    evaluate.add_argument("--reference", required=True, help="Reference transcript")
    evaluate.add_argument("--hypothesis", required=True, help="Hypothesis transcript")

    dataset_eval = sub.add_parser("evaluate-dataset", help="Evaluate WER/CER for a dataset")
    dataset_eval.add_argument("--audio-dir", required=True, help="Directory containing dataset audio files")
    dataset_eval.add_argument("--mapping", required=True, help="JSON/YAML/CSV transcript mapping file")
    dataset_eval.add_argument("--config", help="JSON file with config overrides")
    dataset_eval.add_argument("--output", help="Write results JSON to this path")

    visualize = sub.add_parser("visualize", help="Generate plots from result files")
    visualize.add_argument("--input", required=True, help="Input JSON result file")
    visualize.add_argument("--output-dir", default="plots", help="Directory for generated plots")
    visualize.add_argument(
        "--kind",
        choices=["stage-latency", "chunk-boundaries", "config-comparison", "all"],
        default="all",
        help="Plot type",
    )

    return parser


def main() -> None:
    known_commands = {"transcribe", "compare", "experiments", "evaluate", "visualize"}
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in known_commands:
        legacy = argparse.ArgumentParser(add_help=True)
        legacy.add_argument("audio")
        legacy.add_argument("--config")
        legacy.add_argument("--compare-config")
        legacy_args = legacy.parse_args()
        config_a = _load_overrides(legacy_args.config)
        if legacy_args.compare_config:
            config_b = _load_overrides(legacy_args.compare_config)
            output = compare_configs(legacy_args.audio, config_a, config_b)
        else:
            output = STTService().transcribe(legacy_args.audio, config_a)
        print(json.dumps(output, indent=2))
        return

    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "transcribe":
        output = STTService().transcribe(args.audio, _load_overrides(args.config))
        print(json.dumps(output, indent=2))
        return

    if args.command == "compare":
        output = compare_configs(args.audio, _load_overrides(args.config), _load_overrides(args.compare_config))
        print(json.dumps(output, indent=2))
        return

    if args.command == "experiments":
        output = run_experiments_from_spec(args.spec)
        print(json.dumps(output, indent=2))
        return

    if args.command == "evaluate":
        output = compute_wer_cer(reference=args.reference, hypothesis=args.hypothesis)
        print(json.dumps(output, indent=2))
        return

    if args.command == "evaluate-dataset":
        output = evaluate_dataset(
            audio_dir=args.audio_dir,
            transcript_map_file=args.mapping,
            config_overrides=_load_overrides(args.config),
            output_file=args.output,
        )
        print(json.dumps(output, indent=2))
        return

    if args.command == "visualize":
        created: list[str] = []
        if args.kind in ("stage-latency", "all"):
            created.append(str(plot_stage_latency_breakdown(args.input, args.output_dir)))
        if args.kind in ("chunk-boundaries", "all"):
            created.append(str(plot_chunk_boundaries(args.input, args.output_dir)))
        if args.kind in ("config-comparison", "all"):
            created.append(str(plot_config_comparison(args.input, args.output_dir)))
        print(json.dumps({"plots": created}, indent=2))
        return


if __name__ == "__main__":
    main()
