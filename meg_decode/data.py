"""PyTorch datasets for lazy MEG window loading."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from meg_decode.manifest import ManifestRecord
from meg_decode.preprocessing import (
    compute_window_starts,
    downsampled_length,
    extract_window,
    preprocess_recording,
    window_geometry,
)


@dataclass(frozen=True)
class DataConfig:
    sample_rate: float = 2034.0
    downsample_factor: int = 8
    window_seconds: float = 2.0
    stride_seconds: float = 1.0
    normalization: str = "zscore"
    clip_value: float | None = 6.0
    cache_size: int = 2

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "DataConfig":
        mapping = dict(mapping or {})
        return cls(**mapping)

    @property
    def window_size(self) -> int:
        return window_geometry(
            sample_rate=self.sample_rate,
            downsample_factor=self.downsample_factor,
            window_seconds=self.window_seconds,
            stride_seconds=self.stride_seconds,
        )[0]

    @property
    def stride_size(self) -> int:
        return window_geometry(
            sample_rate=self.sample_rate,
            downsample_factor=self.downsample_factor,
            window_seconds=self.window_seconds,
            stride_seconds=self.stride_seconds,
        )[1]

    @property
    def effective_sample_rate(self) -> float:
        return window_geometry(
            sample_rate=self.sample_rate,
            downsample_factor=self.downsample_factor,
            window_seconds=self.window_seconds,
            stride_seconds=self.stride_seconds,
        )[2]


def read_h5_shape(path: str | Path, dataset_name: str | None = None) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        key = dataset_name or _single_dataset_name(handle)
        return tuple(handle[key].shape)  # type: ignore[return-value]


def load_h5_matrix(record: ManifestRecord) -> np.ndarray:
    with h5py.File(record.path, "r") as handle:
        key = record.dataset_name if record.dataset_name in handle else _single_dataset_name(handle)
        return handle[key][()]


def _single_dataset_name(handle: h5py.File) -> str:
    keys = list(handle.keys())
    if len(keys) != 1:
        raise ValueError(f"Expected exactly one dataset, found {keys}.")
    return keys[0]


class RecordingCache:
    """Small LRU cache for preprocessed recordings."""

    def __init__(self, data_config: DataConfig, max_size: int | None = None) -> None:
        self.data_config = data_config
        self.max_size = data_config.cache_size if max_size is None else max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, record: ManifestRecord) -> np.ndarray:
        key = record.path
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

        raw = load_h5_matrix(record)
        processed = preprocess_recording(
            raw,
            downsample_factor=self.data_config.downsample_factor,
            normalization=self.data_config.normalization,
            clip_value=self.data_config.clip_value,
        )

        if self.max_size > 0:
            self._cache[key] = processed
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
        return processed


class MEGWindowDataset(Dataset):
    """Sliding-window dataset over HDF5 recordings.

    Each item is ``(x, label, record_index)`` where ``x`` has shape
    ``[1, channels, time]`` for EEGNet-style 2D convolutions.
    """

    def __init__(
        self,
        records: list[ManifestRecord],
        data_config: DataConfig,
        *,
        max_windows: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("MEGWindowDataset requires at least one record.")
        self.records = list(records)
        self.data_config = data_config
        self.window_size = data_config.window_size
        self.stride_size = data_config.stride_size
        self.cache = RecordingCache(data_config)
        self.index: list[tuple[int, int]] = []

        for record_index, record in enumerate(self.records):
            _, raw_times = read_h5_shape(record.path, record.dataset_name)
            n_times = downsampled_length(raw_times, data_config.downsample_factor)
            starts = compute_window_starts(n_times, self.window_size, self.stride_size)
            self.index.extend((record_index, start) for start in starts)

        if max_windows is not None:
            self.index = _round_robin_limit(self.index, len(self.records), int(max_windows))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record_index, start = self.index[item]
        record = self.records[record_index]
        recording = self.cache.get(record)
        window = extract_window(recording, start, self.window_size)
        x = torch.from_numpy(window).unsqueeze(0)
        y = torch.tensor(record.label, dtype=torch.long)
        idx = torch.tensor(record_index, dtype=torch.long)
        return x, y, idx


def _round_robin_limit(
    index: list[tuple[int, int]], n_records: int, max_windows: int
) -> list[tuple[int, int]]:
    """Limit windows without accidentally dropping later recordings/classes."""

    if max_windows >= len(index):
        return index
    by_record: list[list[tuple[int, int]]] = [[] for _ in range(n_records)]
    for item in index:
        by_record[item[0]].append(item)

    positions = [0 for _ in range(n_records)]
    limited: list[tuple[int, int]] = []
    while len(limited) < max_windows:
        added = False
        for record_index, record_items in enumerate(by_record):
            pos = positions[record_index]
            if pos < len(record_items):
                limited.append(record_items[pos])
                positions[record_index] += 1
                added = True
                if len(limited) >= max_windows:
                    break
        if not added:
            break
    return limited
