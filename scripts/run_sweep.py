"""Run hyperparameter sweeps from the project config."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meg_decode.manifest import discover_records
from meg_decode.splits import cross_train_subjects
from meg_decode.trainer import train_experiment
from meg_decode.utils import load_yaml, save_csv, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Base YAML config.")
    parser.add_argument("--output-dir", default="outputs/sweeps")
    parser.add_argument("--name", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip trials already listed in sweep_results.csv for this sweep name.",
    )
    parser.add_argument("--skip-reruns", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.config)
    sweep_cfg = base_config.get("sweep", {})
    seeds = sweep_cfg.get("rerun_best_seeds", [0, 1, 2])
    expanded_grid = _uses_expanded_grid(base_config)
    replicate_params = set(sweep_cfg.get("replicate_params", []))

    sweep_name = args.name or f"sweep_{timestamp()}"
    sweep_dir = Path(args.output_dir) / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    folds = _fold_values(base_config)
    rows = _load_existing_rows(sweep_dir / "sweep_results.csv") if args.resume else []
    completed_trials = {row["run_name"] for row in rows}
    run_count = 0
    for trial_params, val_subject in itertools.product(_trial_params(base_config), folds):
        run_name = _run_name_for_trial(trial_params, val_subject, expanded_grid)
        if args.resume and run_name in completed_trials:
            print(f"Skipping completed trial: {run_name}")
            continue
        if args.max_runs is not None and run_count >= args.max_runs:
            break

        config = _config_for_trial(base_config, trial_params, val_subject)
        if args.smoke:
            config.setdefault("training", {})["max_epochs"] = 1
            config.setdefault("debug", {})["records_per_class"] = 1
            config.setdefault("debug", {})["max_windows_per_split"] = 32
        result = train_experiment(config, sweep_dir / run_name)
        rows.append(_result_row(run_name, trial_params, val_subject, result, expanded_grid))
        save_csv(rows, sweep_dir / "sweep_results.csv")
        run_count += 1

    best = _select_best(rows, replicate_params=replicate_params)
    if best is not None and not args.skip_reruns and seeds:
        rerun_path = sweep_dir / "best_seed_reruns.csv"
        rerun_rows = _load_existing_rows(rerun_path) if args.resume else []
        completed_reruns = {row["run_name"] for row in rerun_rows}
        best_params = _params_from_row(best)
        for seed in seeds:
            for val_subject in folds:
                config = _config_for_trial(base_config, best_params, val_subject)
                config.setdefault("training", {})["seed"] = int(seed)
                run_name = f"best_seed{seed}" + (f"_val{val_subject}" if val_subject else "")
                if args.resume and run_name in completed_reruns:
                    print(f"Skipping completed rerun: {run_name}")
                    continue
                result = train_experiment(config, sweep_dir / "reruns" / run_name)
                val_metrics = result["validation"]["file"] if result["validation"] else {}
                row: dict[str, Any] = {
                    "run_name": run_name,
                    "seed": seed,
                    "val_subject": val_subject or "",
                }
                if expanded_grid:
                    row["params_json"] = _params_json(best_params)
                    row.update(_flat_param_columns(best_params))
                row.update(
                    {
                        "val_accuracy": val_metrics.get("accuracy", ""),
                        "val_balanced_accuracy": val_metrics.get("balanced_accuracy", ""),
                        "val_macro_f1": val_metrics.get("macro_f1", ""),
                    }
                )
                rerun_rows.append(row)
                save_csv(rerun_rows, rerun_path)

    print(f"Sweep complete: {sweep_dir}")


def _fold_values(config: dict) -> list[str | None]:
    if config.get("split", {}).get("protocol", "intra") != "cross":
        return [None]
    if config.get("sweep", {}).get("cross_folds", "all") != "all":
        return [config.get("split", {}).get("val_subject")]
    records = discover_records(config.get("data_root", "Final Project data/Final Project data"))
    return cross_train_subjects(records)


def _uses_expanded_grid(config: dict[str, Any]) -> bool:
    sweep_cfg = config.get("sweep", {})
    return any(key in sweep_cfg for key in ("data", "model_params", "training", "replicate_params"))


def _trial_params(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    options = _parameter_options(base_config)
    paths = [path for path, _ in options]
    values = [values for _, values in options]
    return [dict(zip(paths, combo, strict=True)) for combo in itertools.product(*values)]


def _parameter_options(base_config: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    sweep_cfg = base_config.get("sweep", {})
    options: list[tuple[str, list[Any]]] = [
        ("training.learning_rate", _as_options(sweep_cfg.get("learning_rates", [0.001, 0.0003]))),
        ("training.weight_decay", _as_options(sweep_cfg.get("weight_decays", [0.0001, 0.001]))),
        ("model.params.dropout", _as_options(sweep_cfg.get("dropouts", [0.25, 0.5]))),
    ]

    for section, prefix in (
        ("data", "data"),
        ("model_params", "model.params"),
        ("training", "training"),
    ):
        for name, values in sweep_cfg.get(section, {}).items():
            path = f"{prefix}.{name}"
            if path not in {existing_path for existing_path, _ in options}:
                options.append((path, _as_options(values)))
    return options


def _as_options(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _config_for_trial(
    base_config: dict[str, Any], trial_params: dict[str, Any], val_subject: str | None
) -> dict[str, Any]:
    config = deepcopy(base_config)
    for path, value in trial_params.items():
        _set_path(config, path, value)
    if val_subject is not None:
        config.setdefault("split", {})["val_subject"] = str(val_subject)
    config.pop("sweep", None)
    return config


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _run_name_for_trial(
    trial_params: dict[str, Any], val_subject: str | None, expanded_grid: bool
) -> str:
    if not expanded_grid:
        lr = trial_params["training.learning_rate"]
        weight_decay = trial_params["training.weight_decay"]
        dropout = trial_params["model.params.dropout"]
        return (
            f"lr{lr}_wd{weight_decay}_drop{dropout}"
            + (f"_val{val_subject}" if val_subject else "")
        ).replace(".", "p")

    parts = [
        f"{_short_name(path)}{_safe_value(value)}"
        for path, value in sorted(trial_params.items())
    ]
    if val_subject:
        parts.append(f"val{val_subject}")
    return "_".join(parts)


def _short_name(path: str) -> str:
    return {
        "training.learning_rate": "lr",
        "training.weight_decay": "wd",
        "training.seed": "seed",
        "training.batch_size": "bs",
        "model.params.dropout": "drop",
        "model.params.temporal_filters": "tf",
        "model.params.depth_multiplier": "dm",
        "model.params.separable_filters": "sf",
        "model.params.temporal_kernel": "tk",
        "model.params.separable_kernel": "sk",
        "model.params.pool1": "p1",
        "model.params.pool2": "p2",
        "model.params.tcn_dilations": "dil",
        "data.window_seconds": "win",
        "data.stride_seconds": "stride",
        "data.downsample_factor": "down",
        "data.clip_value": "clip",
    }.get(path, path.split(".")[-1])


def _safe_value(value: Any) -> str:
    if isinstance(value, list):
        raw = "-".join(str(item) for item in value)
    else:
        raw = str(value)
    raw = raw.replace(".", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9pm]+", "", raw)


def _result_row(
    run_name: str,
    trial_params: dict[str, Any],
    val_subject: str | None,
    result: dict[str, Any],
    expanded_grid: bool,
) -> dict[str, Any]:
    val_metrics = result["validation"]["file"] if result["validation"] else {}
    if not expanded_grid:
        row: dict[str, Any] = {
            "run_name": run_name,
            "learning_rate": trial_params["training.learning_rate"],
            "weight_decay": trial_params["training.weight_decay"],
            "dropout": trial_params["model.params.dropout"],
            "val_subject": val_subject or "",
            "best_epoch": result["best_epoch"],
        }
    else:
        row = {
            "run_name": run_name,
            "params_json": _params_json(trial_params),
            **_flat_param_columns(trial_params),
            "val_subject": val_subject or "",
            "best_epoch": result["best_epoch"],
        }
    row.update(
        {
            "val_accuracy": val_metrics.get("accuracy", ""),
            "val_balanced_accuracy": val_metrics.get("balanced_accuracy", ""),
            "val_macro_f1": val_metrics.get("macro_f1", ""),
        }
    )
    return row


def _params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _flat_param_columns(params: dict[str, Any]) -> dict[str, Any]:
    return {_short_name(path): _csv_value(value) for path, value in sorted(params.items())}


def _csv_value(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, list) else value


def _params_from_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("params_json"):
        return json.loads(row["params_json"])
    return {
        "training.learning_rate": float(row["learning_rate"]),
        "training.weight_decay": float(row["weight_decay"]),
        "model.params.dropout": float(row["dropout"]),
    }


def _load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_best(rows: list[dict], replicate_params: set[str] | None = None) -> dict | None:
    numeric = [row for row in rows if row["val_balanced_accuracy"] != ""]
    if not numeric:
        return None

    grouped: dict[str | tuple[float, float, float], list[dict]] = {}
    for row in numeric:
        key = _selection_key(row, replicate_params or set())
        grouped.setdefault(key, []).append(row)

    best_key = max(
        grouped,
        key=lambda key: mean(float(row["val_balanced_accuracy"]) for row in grouped[key]),
    )
    representative = grouped[best_key][0].copy()
    representative["val_subject"] = ""
    return representative


def _selection_key(row: dict[str, Any], replicate_params: set[str]) -> str | tuple[float, float, float]:
    if row.get("params_json"):
        params = json.loads(row["params_json"])
        for param in replicate_params:
            params.pop(param, None)
        return _params_json(params)
    return (
        float(row["learning_rate"]),
        float(row["weight_decay"]),
        float(row["dropout"]),
    )


if __name__ == "__main__":
    main()
