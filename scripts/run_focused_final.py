"""Run final cross-subject models selected from the focused sweeps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("configs/eegnet_cross_focused_final.yaml", "eegnet_cross_focused_final"),
    ("configs/tcn_cross_focused_final.yaml", "tcn_cross_focused_final"),
]


def main() -> None:
    for config, run_name in RUNS:
        command = [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            config,
            "--run-name",
            run_name,
        ]
        print("Running: " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    summary_command = [sys.executable, "scripts/summarize_results.py"]
    print("Running: " + " ".join(summary_command), flush=True)
    subprocess.run(summary_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
