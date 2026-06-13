"""Manifest discovery and filename parsing."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from meg_decode.constants import TASK_TO_CLASS, TASK_TO_LABEL

FILENAME_RE = re.compile(
    r"^(?P<task>rest|task_motor|task_story_math|task_working_memory)_"
    r"(?P<subject>\d{6})_(?P<chunk>\d+)\.h5$"
)


@dataclass(frozen=True)
class ManifestRecord:
    """One HDF5 recording with assignment metadata parsed from path and name."""

    path: str
    protocol: str
    split: str
    task: str
    class_name: str
    label: int
    subject: str
    chunk: int

    @property
    def dataset_name(self) -> str:
        return f"{self.task}_{self.subject}"


def resolve_data_root(data_root: str | Path) -> Path:
    """Resolve either the inner data root or the provided outer extracted folder."""

    root = Path(data_root)
    if (root / "Intra").is_dir() and (root / "Cross").is_dir():
        return root

    nested = root / "Final Project data"
    if (nested / "Intra").is_dir() and (nested / "Cross").is_dir():
        return nested

    raise FileNotFoundError(
        f"Could not find Intra/Cross folders under {root!s} or {nested!s}."
    )


def parse_h5_filename(filename: str | Path) -> tuple[str, str, int]:
    """Parse task, subject, and chunk from an assignment HDF5 filename."""

    name = Path(filename).name
    match = FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected HDF5 filename: {name}")
    task = match.group("task")
    subject = match.group("subject")
    chunk = int(match.group("chunk"))
    return task, subject, chunk


def discover_records(data_root: str | Path) -> list[ManifestRecord]:
    """Find all assignment HDF5 files and return a sorted manifest."""

    root = resolve_data_root(data_root)
    records: list[ManifestRecord] = []
    for path in sorted(root.rglob("*.h5")):
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            continue
        protocol, split = relative.parts[0], relative.parts[1]
        task, subject, chunk = parse_h5_filename(path.name)
        records.append(
            ManifestRecord(
                path=str(path),
                protocol=protocol,
                split=split,
                task=task,
                class_name=TASK_TO_CLASS[task],
                label=TASK_TO_LABEL[task],
                subject=subject,
                chunk=chunk,
            )
        )
    if not records:
        raise FileNotFoundError(f"No .h5 files found under {root!s}.")
    return records


def write_manifest_csv(records: Iterable[ManifestRecord], output_path: str | Path) -> None:
    """Write a manifest CSV for auditability."""

    records = list(records)
    if not records:
        raise ValueError("Cannot write an empty manifest.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest_csv(path: str | Path) -> list[ManifestRecord]:
    """Load a manifest CSV created by :func:`write_manifest_csv`."""

    records: list[ManifestRecord] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["label"] = int(row["label"])
            row["chunk"] = int(row["chunk"])
            records.append(ManifestRecord(**row))
    return records

