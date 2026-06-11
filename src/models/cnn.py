import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.shortcut: nn.Module
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.act(out)
        return out


class ResNet1DEncoder(nn.Module):
    """
    ResNet-style encoder for inputs with 248 features.

    Input:
      - [B, T, 248] or [T, 248]
    Output:
      - [B, T, latent_dim] or [T, latent_dim]
    """

    def __init__(self, input_features: int = 248, base_channels: int = 128, latent_dim: int = 64):
        super().__init__()
        self.input_features = input_features

        self.stem = nn.Sequential(
            nn.Conv1d(input_features, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels),
            ResidualBlock1D(base_channels, base_channels),
            ResidualBlock1D(base_channels, latent_dim),
            ResidualBlock1D(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        if x.dim() != 3 or x.size(-1) != self.input_features:
            raise ValueError(
                f"Expected [B, T, {self.input_features}] or [T, {self.input_features}], got {tuple(x.shape)}."
            )

        x = x.permute(0, 2, 1)  # [B, T, F] -> [B, F, T]
        z = self.blocks(self.stem(x))
        z = z.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]

        if squeeze_batch:
            z = z.squeeze(0)
        return z


class ResNet1DDecoder(nn.Module):
    """
    ResNet-style decoder that mirrors the encoder.

    Input:
      - [B, T, latent_dim] or [T, latent_dim]
    Output:
      - [B, T, 248] or [T, 248]
    """

    def __init__(self, output_features: int = 248, base_channels: int = 128, latent_dim: int = 64):
        super().__init__()
        self.output_features = output_features

        self.stem = nn.Sequential(
            nn.Conv1d(latent_dim, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels),
            ResidualBlock1D(base_channels, base_channels),
        )

        self.head = nn.Conv1d(base_channels, output_features, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze_batch = True

        if z.dim() != 3:
            raise ValueError(f"Expected [B, T, C] or [T, C], got {tuple(z.shape)}.")

        z = z.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
        x_hat = self.blocks(self.stem(z))
        x_hat = self.head(x_hat)
        x_hat = x_hat.permute(0, 2, 1)  # [B, F, T] -> [B, T, F]

        if squeeze_batch:
            x_hat = x_hat.squeeze(0)
        return x_hat
