"""Signal preprocessing and window geometry."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly


def downsample_recording(recording: np.ndarray, factor: int) -> np.ndarray:
    """Anti-aliased downsampling along time."""

    recording = np.asarray(recording, dtype=np.float32)
    if factor <= 1:
        return recording
    return resample_poly(recording, up=1, down=int(factor), axis=-1).astype(
        np.float32, copy=False
    )


def normalize_timewise_zscore(recording: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize each sensor using its own time-axis mean and standard deviation."""

    mean = recording.mean(axis=-1, keepdims=True)
    std = recording.std(axis=-1, keepdims=True)
    return (recording - mean) / np.maximum(std, eps)


def normalize_timewise_minmax(recording: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize each sensor to [0, 1] over time."""

    low = recording.min(axis=-1, keepdims=True)
    high = recording.max(axis=-1, keepdims=True)
    return (recording - low) / np.maximum(high - low, eps)


def preprocess_recording(
    recording: np.ndarray,
    *,
    downsample_factor: int = 8,
    normalization: str = "zscore",
    clip_value: float | None = 6.0,
) -> np.ndarray:
    """Apply the assignment preprocessing pipeline to a full recording."""

    recording = downsample_recording(recording, downsample_factor)

    if normalization == "zscore":
        recording = normalize_timewise_zscore(recording)
    elif normalization == "minmax":
        recording = normalize_timewise_minmax(recording)
    elif normalization in ("none", None):
        pass
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if clip_value is not None:
        recording = np.clip(recording, -float(clip_value), float(clip_value))

    return recording.astype(np.float32, copy=False)


def downsampled_length(raw_length: int, factor: int) -> int:
    """Length produced by scipy.signal.resample_poly with up=1."""

    if factor <= 1:
        return int(raw_length)
    return int(math.ceil(raw_length / factor))


def window_geometry(
    *,
    sample_rate: float,
    downsample_factor: int,
    window_seconds: float,
    stride_seconds: float,
) -> tuple[int, int, float]:
    """Return downsampled window size, stride size, and effective sample rate."""

    effective_rate = sample_rate / downsample_factor
    window_size = max(1, int(round(window_seconds * effective_rate)))
    stride_size = max(1, int(round(stride_seconds * effective_rate)))
    return window_size, stride_size, effective_rate


def compute_window_starts(n_times: int, window_size: int, stride_size: int) -> list[int]:
    """Compute deterministic sliding-window starts."""

    if n_times <= 0:
        raise ValueError("n_times must be positive.")
    if window_size <= 0 or stride_size <= 0:
        raise ValueError("window_size and stride_size must be positive.")
    if n_times <= window_size:
        return [0]
    return list(range(0, n_times - window_size + 1, stride_size))


def extract_window(recording: np.ndarray, start: int, window_size: int) -> np.ndarray:
    """Slice a fixed-size window, zero-padding on the right only if needed."""

    end = start + window_size
    window = recording[:, start:end]
    if window.shape[-1] == window_size:
        return window

    padded = np.zeros((recording.shape[0], window_size), dtype=recording.dtype)
    padded[:, : window.shape[-1]] = window
    return padded

