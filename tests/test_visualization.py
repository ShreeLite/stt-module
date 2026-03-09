from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stt_module.visualization.plots import (
    plot_chunk_boundaries,
    plot_config_comparison,
    plot_stage_latency_breakdown,
)


class TestVisualization(unittest.TestCase):
    def test_plot_generators_create_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            exp_file = d / "exp.json"
            exp_file.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "experiment": "baseline",
                                "latency_ms": 100.0,
                                "confidence": 0.8,
                                "stage_latencies_ms": {"preprocessing": 2.0, "recognition": 90.0},
                            },
                            {
                                "experiment": "vad",
                                "latency_ms": 130.0,
                                "confidence": 0.75,
                                "stage_latencies_ms": {"preprocessing": 3.0, "recognition": 110.0},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            single_file = d / "single.json"
            single_file.write_text(
                json.dumps(
                    {
                        "metrics": {"audio_duration_s": 10.0},
                        "debug": {
                            "waveform_envelope": [0.1, 0.4, 0.2, 0.3],
                            "chunk_boundaries": [
                                {"start_s": 0.0, "end_s": 3.0},
                                {"start_s": 2.5, "end_s": 6.0},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            out_dir = d / "plots"
            self.assertTrue(plot_stage_latency_breakdown(exp_file, out_dir).exists())
            self.assertTrue(plot_config_comparison(exp_file, out_dir).exists())
            self.assertTrue(plot_chunk_boundaries(single_file, out_dir).exists())


if __name__ == "__main__":
    unittest.main()
