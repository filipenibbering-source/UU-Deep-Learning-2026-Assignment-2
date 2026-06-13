"""Training orchestration for assignment experiments."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from meg_decode.constants import CLASS_NAMES
from meg_decode.data import DataConfig, MEGWindowDataset, read_h5_shape
from meg_decode.evaluation import evaluate_model, write_prediction_csv
from meg_decode.manifest import ManifestRecord, discover_records, write_manifest_csv
from meg_decode.metrics import plot_confusion_matrix, write_confusion_csv
from meg_decode.models import count_parameters, make_model
from meg_decode.splits import DataSplit, build_split
from meg_decode.utils import get_device, save_csv, save_json, set_seed


def train_experiment(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = config.get("training", {})
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(training_cfg.get("device", "auto"))

    records = discover_records(config.get("data_root", "Final Project data/Final Project data"))
    split = build_split(records, config.get("split", {"protocol": "intra"}))
    split = _apply_debug_limits(split, config.get("debug", {}))
    write_manifest_csv(records, run_dir / "manifest.csv")

    data_config = DataConfig.from_mapping(config.get("data", {}))
    max_windows = config.get("debug", {}).get("max_windows_per_split")
    train_dataset = MEGWindowDataset(split.train, data_config, max_windows=max_windows)
    val_dataset = (
        MEGWindowDataset(split.val, data_config, max_windows=max_windows) if split.val else None
    )
    test_datasets = {
        name: MEGWindowDataset(group_records, data_config, max_windows=max_windows)
        for name, group_records in split.test_groups.items()
    }

    loader_kwargs = {
        "batch_size": int(training_cfg.get("batch_size", 16)),
        "num_workers": int(training_cfg.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    eval_loader_kwargs = dict(loader_kwargs)
    eval_loader_kwargs["shuffle"] = False
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs) if val_dataset else None
    test_loaders = {
        name: DataLoader(dataset, **eval_loader_kwargs)
        for name, dataset in test_datasets.items()
    }

    model = _build_model(config, split.train, data_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 0.001)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0001)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=int(training_cfg.get("lr_patience", 5))
    )
    criterion = nn.CrossEntropyLoss()

    use_amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_epochs = int(training_cfg.get("max_epochs", 100))
    patience = int(training_cfg.get("patience", 15))

    metadata = {
        "class_names": CLASS_NAMES,
        "data_config": data_config.__dict__,
        "split": split.description,
        "model_parameters": count_parameters(model),
        "device": str(device),
        "seed": seed,
    }
    save_json({"config": config, "metadata": metadata}, run_dir / "run_config.json")

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_stats = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        row: dict[str, Any] = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}
        score = -train_stats["loss"]
        if val_loader is not None:
            val_results = evaluate_model(model, val_loader, split.val, device)
            score = float(val_results["file"]["balanced_accuracy"])
            row.update(_flatten_metrics("val_window", val_results["window"]))
            row.update(_flatten_metrics("val_file", val_results["file"]))

        scheduler.step(score)
        row["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        save_csv(history, run_dir / "history.csv")

        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(model, config, metadata, run_dir / "best.pt", epoch, best_score)
        else:
            epochs_without_improvement += 1

        _save_checkpoint(model, config, metadata, run_dir / "last.pt", epoch, score)

        if val_loader is not None and epochs_without_improvement >= patience:
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_results: dict[str, Any] = {
        "metadata": metadata,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "validation": None,
        "tests": {},
    }

    if val_loader is not None:
        val_results = evaluate_model(model, val_loader, split.val, device)
        _write_eval_outputs(run_dir / "validation", val_results)
        final_results["validation"] = {
            "window": val_results["window"],
            "file": val_results["file"],
        }

    for group_name, loader in test_loaders.items():
        test_results = evaluate_model(model, loader, test_datasets[group_name].records, device)
        _write_eval_outputs(run_dir / f"test_{group_name}", test_results)
        final_results["tests"][group_name] = {
            "window": test_results["window"],
            "file": test_results["file"],
        }

    save_json(final_results, run_dir / "metrics.json")
    return final_results


def _train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    progress = tqdm(loader, leave=False, desc="train")
    for x, y, _ in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(y.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        total_examples += batch_size
        progress.set_postfix(loss=total_loss / max(total_examples, 1))

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def _build_model(
    config: dict[str, Any], train_records: list[ManifestRecord], data_config: DataConfig
) -> nn.Module:
    model_cfg = deepcopy(config.get("model", {}))
    name = model_cfg.pop("name", "eegnet")
    params = model_cfg.pop("params", {})
    n_channels, _ = read_h5_shape(train_records[0].path, train_records[0].dataset_name)
    params.setdefault("n_channels", n_channels)
    params.setdefault("n_classes", len(CLASS_NAMES))
    params.setdefault("n_times", data_config.window_size)
    params.pop("n_times", None)
    return make_model(name, **params)


def _write_eval_outputs(output_dir: Path, results: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_csv(results["file_predictions"], output_dir / "file_predictions.csv")
    for level in ("window", "file"):
        matrix = torch.tensor(results[level]["confusion_matrix"]).numpy()
        write_confusion_csv(matrix, CLASS_NAMES, output_dir / f"{level}_confusion_matrix.csv")
        plot_confusion_matrix(
            matrix,
            CLASS_NAMES,
            output_dir / f"{level}_confusion_matrix.png",
            title=f"{level.capitalize()} confusion matrix",
        )


def _save_checkpoint(
    model: nn.Module,
    config: dict[str, Any],
    metadata: dict[str, Any],
    path: Path,
    epoch: int,
    score: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "metadata": metadata,
            "epoch": epoch,
            "score": score,
        },
        path,
    )


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall"):
        flat[f"{prefix}_{key}"] = metrics[key]
    return flat


def _apply_debug_limits(split: DataSplit, debug_cfg: dict[str, Any]) -> DataSplit:
    records_per_class = debug_cfg.get("records_per_class")
    if records_per_class is None:
        return split
    limit = int(records_per_class)
    return DataSplit(
        train=_limit_records_per_class(split.train, limit),
        val=_limit_records_per_class(split.val, limit),
        test_groups={
            name: _limit_records_per_class(records, limit)
            for name, records in split.test_groups.items()
        },
        description={**split.description, "debug_records_per_class": limit},
    )


def _limit_records_per_class(
    records: list[ManifestRecord], records_per_class: int
) -> list[ManifestRecord]:
    counts: dict[int, int] = {}
    limited: list[ManifestRecord] = []
    for record in records:
        count = counts.get(record.label, 0)
        if count < records_per_class:
            limited.append(record)
            counts[record.label] = count + 1
    return limited

