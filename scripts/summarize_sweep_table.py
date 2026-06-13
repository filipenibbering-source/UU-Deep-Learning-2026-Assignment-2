"""Create an audit-style Markdown table for one sweep directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True, help="Directory containing sweep_results.csv.")
    parser.add_argument("--output", default=None, help="Markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    rows = _load_csv(sweep_dir / "sweep_results.csv")
    if not rows:
        raise ValueError(f"No rows found in {sweep_dir / 'sweep_results.csv'}")

    output = Path(args.output) if args.output else sweep_dir / "sweep_table.md"
    lines = [f"# Sweep Table: {sweep_dir.name}", ""]
    lines.extend(_table_for_rows(sweep_dir, rows, reruns=False))

    rerun_path = sweep_dir / "best_seed_reruns.csv"
    if rerun_path.exists():
        rerun_rows = _load_csv(rerun_path)
        if rerun_rows:
            lines.extend(["", "## Seed Reruns", ""])
            lines.extend(_table_for_rows(sweep_dir / "reruns", rerun_rows, reruns=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _table_for_rows(base_dir: Path, rows: list[dict[str, str]], *, reruns: bool) -> list[str]:
    excluded = {
        "run_name",
        "params_json",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
    }
    param_columns = [column for column in rows[0] if column not in excluded]
    metric_columns = _metric_columns(base_dir, rows)
    columns = [*param_columns, *metric_columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        metrics = _load_metrics(base_dir / row["run_name"] / "metrics.json")
        values: dict[str, Any] = {column: row.get(column, "") for column in param_columns}
        if metrics.get("validation"):
            values["val_bal"] = metrics["validation"]["file"]["balanced_accuracy"]
            values["val_f1"] = metrics["validation"]["file"]["macro_f1"]
        for group, group_metrics in metrics["tests"].items():
            values[f"{group}_bal"] = group_metrics["file"]["balanced_accuracy"]
            values[f"{group}_f1"] = group_metrics["file"]["macro_f1"]
        lines.append("| " + " | ".join(_format(values.get(column, "")) for column in columns) + " |")
    return lines


def _metric_columns(base_dir: Path, rows: list[dict[str, str]]) -> list[str]:
    metrics = _load_metrics(base_dir / rows[0]["run_name"] / "metrics.json")
    columns = ["val_bal", "val_f1"] if metrics.get("validation") else []
    for group in metrics["tests"]:
        columns.extend([f"{group}_bal", f"{group}_f1"])
    return columns


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


if __name__ == "__main__":
    main()
