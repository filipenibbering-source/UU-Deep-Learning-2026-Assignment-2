"""Metrics without a scikit-learn dependency."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        matrix[true_label, pred_label] += 1
    return matrix


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total else 0.0

    recalls: list[float] = []
    precisions: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []
    for class_index in range(n_classes):
        tp = float(cm[class_index, class_index])
        support = int(cm[class_index].sum())
        predicted = int(cm[:, class_index].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        recalls.append(float(recall))
        precisions.append(float(precision))
        f1s.append(float(f1))
        supports.append(support)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "support": total,
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "per_class_f1": f1s,
        "per_class_support": supports,
        "confusion_matrix": cm.tolist(),
    }


def write_confusion_csv(
    matrix: np.ndarray, class_names: tuple[str, ...], output_path: str | Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for class_name, row in zip(class_names, matrix.tolist(), strict=False):
            writer.writerow([class_name, *row])


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: tuple[str, ...],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

