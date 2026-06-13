"""Leak-safe assignment split construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from meg_decode.manifest import ManifestRecord


@dataclass(frozen=True)
class DataSplit:
    train: list[ManifestRecord]
    val: list[ManifestRecord]
    test_groups: dict[str, list[ManifestRecord]]
    description: dict[str, Any]


def _sorted(records: list[ManifestRecord]) -> list[ManifestRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.protocol,
            record.split,
            record.subject,
            record.task,
            record.chunk,
            record.path,
        ),
    )


def records_for_protocol(records: list[ManifestRecord], protocol: str) -> list[ManifestRecord]:
    return [record for record in records if record.protocol.lower() == protocol.lower()]


def cross_train_subjects(records: list[ManifestRecord]) -> list[str]:
    subjects = {
        record.subject
        for record in records
        if record.protocol.lower() == "cross" and record.split == "train"
    }
    return sorted(subjects)


def validate_no_overlap(split: DataSplit) -> None:
    """Raise if the same file appears in more than one logical split."""

    groups = {"train": split.train, "val": split.val}
    groups.update(split.test_groups)
    seen: dict[str, str] = {}
    for group_name, group_records in groups.items():
        for record in group_records:
            previous = seen.get(record.path)
            if previous is not None:
                raise ValueError(
                    f"File leakage detected: {record.path} in {previous} and {group_name}."
                )
            seen[record.path] = group_name


def build_split(records: list[ManifestRecord], config: dict[str, Any]) -> DataSplit:
    """Build the split requested by a run config."""

    protocol = str(config.get("protocol", "intra")).lower()
    if protocol == "intra":
        split = _build_intra_split(records, config)
    elif protocol == "cross":
        split = _build_cross_split(records, config)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    validate_no_overlap(split)
    return split


def _build_intra_split(records: list[ManifestRecord], config: dict[str, Any]) -> DataSplit:
    train_chunks = set(config.get("train_chunks", [1, 2, 3, 4, 5, 6]))
    val_chunks = set(config.get("val_chunks", [7, 8]))

    protocol_records = records_for_protocol(records, "Intra")
    train = [
        record
        for record in protocol_records
        if record.split == "train" and record.chunk in train_chunks
    ]
    val = [
        record for record in protocol_records if record.split == "train" and record.chunk in val_chunks
    ]
    test = [record for record in protocol_records if record.split == "test"]

    split = DataSplit(
        train=_sorted(train),
        val=_sorted(val),
        test_groups={"test": _sorted(test)},
        description={
            "protocol": "intra",
            "train_chunks": sorted(train_chunks),
            "val_chunks": sorted(val_chunks),
            "test_split": "Intra/test",
        },
    )
    _ensure_non_empty(split)
    return split


def _build_cross_split(records: list[ManifestRecord], config: dict[str, Any]) -> DataSplit:
    protocol_records = records_for_protocol(records, "Cross")
    train_pool = [record for record in protocol_records if record.split == "train"]
    subjects = sorted({record.subject for record in train_pool})
    if len(subjects) < 2:
        raise ValueError("Cross-subject validation requires at least two train subjects.")

    final_train = bool(config.get("final_train", False))
    val_subject = config.get("val_subject")
    if val_subject in ("", "null", "None"):
        val_subject = None
    if val_subject is None:
        val_subject = subjects[-1]
    val_subject = str(val_subject)
    if val_subject not in subjects:
        raise ValueError(f"val_subject={val_subject} is not one of {subjects}.")

    if final_train:
        train = train_pool
        val: list[ManifestRecord] = []
    else:
        train = [record for record in train_pool if record.subject != val_subject]
        val = [record for record in train_pool if record.subject == val_subject]

    eval_splits = config.get("eval_splits", ["test1", "test2", "test3"])
    test_groups = {
        split_name: _sorted(
            [record for record in protocol_records if record.split == split_name]
        )
        for split_name in eval_splits
    }

    split = DataSplit(
        train=_sorted(train),
        val=_sorted(val),
        test_groups=test_groups,
        description={
            "protocol": "cross",
            "train_subjects": sorted({record.subject for record in train}),
            "val_subject": None if final_train else val_subject,
            "final_train": final_train,
            "test_splits": list(eval_splits),
        },
    )
    _ensure_non_empty(split, require_val=not final_train)
    return split


def _ensure_non_empty(split: DataSplit, require_val: bool = True) -> None:
    if not split.train:
        raise ValueError("Training split is empty.")
    if require_val and not split.val:
        raise ValueError("Validation split is empty.")
    for name, records in split.test_groups.items():
        if not records:
            raise ValueError(f"Test split {name} is empty.")

