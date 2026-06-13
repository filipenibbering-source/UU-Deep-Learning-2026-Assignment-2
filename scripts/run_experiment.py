"""Run one MEG decoding experiment."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meg_decode.trainer import train_experiment
from meg_decode.utils import load_yaml, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment config.")
    parser.add_argument("--output-dir", default="outputs/runs", help="Base output directory.")
    parser.add_argument("--run-name", default=None, help="Optional run directory name.")
    parser.add_argument("--data-root", default=None, help="Override data root.")
    parser.add_argument("--protocol", choices=["intra", "cross"], default=None)
    parser.add_argument("--model", choices=["eegnet", "eegnet_tcn"], default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny one-epoch sanity check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config = deepcopy(config)

    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.protocol is not None:
        config.setdefault("split", {})["protocol"] = args.protocol
    if args.model is not None:
        config.setdefault("model", {})["name"] = args.model
    if args.epochs is not None:
        config.setdefault("training", {})["max_epochs"] = args.epochs
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    if args.smoke:
        config.setdefault("training", {})["max_epochs"] = 1
        config.setdefault("training", {})["batch_size"] = 8
        config.setdefault("debug", {})["records_per_class"] = 1
        config.setdefault("debug", {})["max_windows_per_split"] = 32

    protocol = config.get("split", {}).get("protocol", "intra")
    model = config.get("model", {}).get("name", "eegnet")
    run_name = args.run_name or f"{protocol}_{model}_{timestamp()}"
    run_dir = Path(args.output_dir) / run_name
    results = train_experiment(config, run_dir)
    print(f"Run complete: {run_dir}")
    print(f"Best epoch: {results['best_epoch']} score={results['best_score']:.4f}")
    for group, metrics in results["tests"].items():
        file_metrics = metrics["file"]
        print(
            f"test/{group}: accuracy={file_metrics['accuracy']:.4f} "
            f"balanced_accuracy={file_metrics['balanced_accuracy']:.4f} "
            f"macro_f1={file_metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
