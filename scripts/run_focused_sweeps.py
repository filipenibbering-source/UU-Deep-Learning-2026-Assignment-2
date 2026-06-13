"""Run the remaining focused sweeps sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REMAINING_SWEEPS = [
    ("eegnet_cross_focused", "configs/eegnet_cross_focused.yaml"),
    ("tcn_intra_focused", "configs/tcn_intra_focused.yaml"),
    ("tcn_cross_focused", "configs/tcn_cross_focused.yaml"),
]

ALL_SWEEPS = [
    ("eegnet_intra_focused", "configs/eegnet_intra_focused.yaml"),
    *REMAINING_SWEEPS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Also run eegnet_intra_focused with --resume before the remaining sweeps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweeps = ALL_SWEEPS if args.include_completed else REMAINING_SWEEPS
    for name, config in sweeps:
        command = [
            sys.executable,
            "scripts/run_sweep.py",
            "--config",
            config,
            "--name",
            name,
            "--resume",
        ]
        print("Running: " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
