"""Model evaluation and file-level aggregation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from meg_decode.constants import CLASS_NAMES
from meg_decode.manifest import ManifestRecord
from meg_decode.metrics import classification_metrics


def predict_dataloader(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_record_indices: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, record_idx in dataloader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.numpy())
            all_record_indices.append(record_idx.numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_record_indices, axis=0),
    )


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def aggregate_file_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    record_indices: np.ndarray,
    records: list[ManifestRecord],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    grouped_probs: dict[int, list[np.ndarray]] = defaultdict(list)
    grouped_labels: dict[int, list[int]] = defaultdict(list)
    for probs, label, record_idx in zip(probabilities, labels, record_indices, strict=False):
        grouped_probs[int(record_idx)].append(probs)
        grouped_labels[int(record_idx)].append(int(label))

    file_labels: list[int] = []
    file_preds: list[int] = []
    rows: list[dict[str, Any]] = []
    for record_idx in sorted(grouped_probs):
        mean_probs = np.mean(np.stack(grouped_probs[record_idx], axis=0), axis=0)
        label_values = set(grouped_labels[record_idx])
        if len(label_values) != 1:
            raise ValueError(f"Mixed labels for record index {record_idx}: {label_values}")
        true_label = grouped_labels[record_idx][0]
        pred_label = int(np.argmax(mean_probs))
        record = records[record_idx]
        file_labels.append(true_label)
        file_preds.append(pred_label)
        row = {
            "path": record.path,
            "protocol": record.protocol,
            "split": record.split,
            "subject": record.subject,
            "chunk": record.chunk,
            "task": record.task,
            "true_label": true_label,
            "true_class": record.class_name,
            "pred_label": pred_label,
            "pred_class": CLASS_NAMES[pred_label],
            "n_windows": len(grouped_probs[record_idx]),
        }
        for class_index, class_name in enumerate(CLASS_NAMES):
            row[f"prob_{class_name}"] = float(mean_probs[class_index])
        rows.append(row)

    return np.array(file_labels), np.array(file_preds), rows


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    records: list[ManifestRecord],
    device: torch.device,
) -> dict[str, Any]:
    logits, labels, record_indices = predict_dataloader(model, dataloader, device)
    probabilities = softmax_np(logits)
    window_preds = np.argmax(probabilities, axis=1)
    file_labels, file_preds, file_rows = aggregate_file_predictions(
        probabilities, labels, record_indices, records
    )
    return {
        "window": classification_metrics(labels, window_preds, len(CLASS_NAMES)),
        "file": classification_metrics(file_labels, file_preds, len(CLASS_NAMES)),
        "file_predictions": file_rows,
    }


def write_prediction_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    if not rows:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

