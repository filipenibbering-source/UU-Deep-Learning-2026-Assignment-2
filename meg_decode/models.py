"""Compact neural models for MEG decoding."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EEGNetStem(nn.Module):
    """EEGNet-style temporal, spatial, and separable convolution feature extractor."""

    def __init__(
        self,
        *,
        n_channels: int = 248,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        separable_filters: int = 32,
        temporal_kernel: int = 63,
        separable_kernel: int = 15,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        depthwise_filters = temporal_filters * depth_multiplier
        self.out_channels = separable_filters

        self.net = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                depthwise_filters,
                kernel_size=(n_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(depthwise_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(dropout),
            nn.Conv2d(
                depthwise_filters,
                depthwise_filters,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=depthwise_filters,
                bias=False,
            ),
            nn.Conv2d(depthwise_filters, separable_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(separable_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGNetClassifier(nn.Module):
    """Compact EEGNet-style classifier."""

    def __init__(self, n_classes: int = 4, **stem_kwargs: Any) -> None:
        super().__init__()
        self.stem = EEGNetStem(**stem_kwargs)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.stem.out_channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stem(x))


class ResidualTCNBlock(nn.Module):
    """Residual dilated temporal block for the improved model."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class EEGNetTCNClassifier(nn.Module):
    """EEGNet front-end followed by lightweight residual temporal convolutions."""

    def __init__(
        self,
        n_classes: int = 4,
        tcn_dilations: list[int] | tuple[int, ...] = (1, 2, 4),
        tcn_dropout: float | None = None,
        **stem_kwargs: Any,
    ) -> None:
        super().__init__()
        stem_dropout = float(stem_kwargs.get("dropout", 0.5))
        self.stem = EEGNetStem(**stem_kwargs)
        dropout = stem_dropout if tcn_dropout is None else float(tcn_dropout)
        self.tcn = nn.Sequential(
            *[
                ResidualTCNBlock(self.stem.out_channels, int(dilation), dropout)
                for dilation in tcn_dilations
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.stem.out_channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x).squeeze(2)
        x = self.tcn(x)
        return self.head(x)


def make_model(name: str, **kwargs: Any) -> nn.Module:
    name = name.lower()
    if name == "eegnet":
        return EEGNetClassifier(**kwargs)
    if name in {"eegnet_tcn", "tcn"}:
        return EEGNetTCNClassifier(**kwargs)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

