"""Generate a compact Markdown summary from experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--output", default="outputs/results_summary.md")
    args = parser.parse_args()

    metrics_files = sorted(Path(args.outputs).rglob("metrics.json"))
    lines = ["# MEG Decoding Results", ""]
    if not metrics_files:
        lines.append("No metrics.json files found.")
    for metrics_file in metrics_files:
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        metadata = metrics.get("metadata", {})
        split = metadata.get("split", {})
        lines.append(f"## {metrics_file.parent}")
        lines.append("")
        lines.append(f"- Protocol: `{split.get('protocol', 'unknown')}`")
        lines.append(f"- Best epoch: `{metrics.get('best_epoch')}`")
        lines.append(f"- Best score: `{metrics.get('best_score'):.4f}`")
        lines.append(f"- Train subjects: `{split.get('train_subjects', '')}`")
        if metrics.get("validation"):
            val = metrics["validation"]["file"]
            lines.append(
                "- Validation file metrics: "
                f"accuracy `{val['accuracy']:.4f}`, "
                f"balanced accuracy `{val['balanced_accuracy']:.4f}`, "
                f"macro-F1 `{val['macro_f1']:.4f}`"
            )
        for group, group_metrics in metrics.get("tests", {}).items():
            file_metrics = group_metrics["file"]
            lines.append(
                f"- Test `{group}` file metrics: "
                f"accuracy `{file_metrics['accuracy']:.4f}`, "
                f"balanced accuracy `{file_metrics['balanced_accuracy']:.4f}`, "
                f"macro-F1 `{file_metrics['macro_f1']:.4f}`"
            )
        lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
