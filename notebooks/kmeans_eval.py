from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from sklearn.cluster import KMeans


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _cluster_to_label_mapping(
    train_clusters: np.ndarray,
    train_labels: np.ndarray,
    n_clusters: int,
) -> dict[int, int]:
    """Map each cluster id to the majority task label in the training set."""
    mapping: dict[int, int] = {}
    values, counts = np.unique(train_labels, return_counts=True)
    fallback_label = int(values[counts.argmax()])

    for cluster_id in range(n_clusters):
        mask = train_clusters == cluster_id
        if not np.any(mask):
            mapping[cluster_id] = fallback_label
            continue

        cluster_labels = train_labels[mask]
        cluster_values, cluster_counts = np.unique(cluster_labels, return_counts=True)
        mapping[cluster_id] = int(cluster_values[cluster_counts.argmax()])

    return mapping


def evaluate_kmeans_on_cross_test_sets(
    train_embeddings: np.ndarray | torch.Tensor,
    train_labels: np.ndarray | torch.Tensor,
    test_sets: Mapping[str, tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]],
    k_values: range | list[int] = range(1, 6),
    random_state: int = 42,
    verbose: bool = True,
) -> dict[int, dict[str, float]]:
    """Fit k-means on train embeddings and evaluate on cross test splits.

    For each k, cluster ids are mapped to task labels using the majority label
    of training points in each cluster. Accuracy is reported on each test set.
    """
    train_x = _to_numpy(train_embeddings)
    train_y = _to_numpy(train_labels)

    test_data = {
        name: (_to_numpy(embeddings), _to_numpy(labels))
        for name, (embeddings, labels) in test_sets.items()
    }

    results: dict[int, dict[str, float]] = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        train_clusters = kmeans.fit_predict(train_x)
        label_mapping = _cluster_to_label_mapping(train_clusters, train_y, k)

        # print('train')

        results[k] = {}
        if verbose:
            print(f"k={k}")

        for test_name, (test_x, test_y) in test_data.items():
            test_clusters = kmeans.predict(test_x)
            predicted_labels = np.array([label_mapping[cluster_id] for cluster_id in test_clusters])
            accuracy = float((predicted_labels == test_y).mean())
            results[k][test_name] = accuracy

            if verbose:
                print(f"  {test_name}: accuracy={accuracy:.4f}")

    if verbose:
        test_names = list(test_data.keys())
        header = " ".join(f"{name:>8}" for name in test_names)
        print("\nSummary (k-means accuracy on cross test sets):")
        print(f"{'k':>3}  {header}")
        for k in k_values:
            row = " ".join(f"{results[k][name]:>8.4f}" for name in test_names)
            print(f"{k:>3}  {row}")

    return results
