"""Factory functions for creating ablated PrismNet variants."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AblationConfig


class IdentitySE(nn.Module):
    """SE block replacement that returns ones (no channel attention)."""

    def forward(self, x):
        b, c, _, _ = x.size()
        return torch.ones(b, c, 1, 1, device=x.device, dtype=x.dtype)


class Conv2dAblated(nn.Module):
    """Conv2d with optional BatchNorm ablation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        relu=True,
        same_padding=False,
        bn=False,
        disable_batchnorm=False,
    ):
        super().__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding=padding
        )

        # Apply BatchNorm ablation
        if bn and not disable_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ResidualBlock1DAblated(nn.Module):
    """ResidualBlock1D with optional skip connection and BatchNorm ablation."""

    def __init__(self, planes, disable_skip=False, disable_batchnorm=False):
        super().__init__()
        self.disable_skip = disable_skip

        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.c2 = nn.Conv1d(
            planes, planes * 2, kernel_size=11, stride=1, padding=5, bias=False
        )
        self.c3 = nn.Conv1d(
            planes * 2, planes * 8, kernel_size=1, stride=1, bias=False
        )

        if disable_batchnorm:
            self.b1 = nn.Identity()
            self.b2 = nn.Identity()
            self.b3 = nn.Identity()
        else:
            self.b1 = nn.BatchNorm1d(planes)
            self.b2 = nn.BatchNorm1d(planes * 2)
            self.b3 = nn.BatchNorm1d(planes * 8)

        # Downsample path (always needed for channel expansion)
        if disable_batchnorm:
            self.downsample = nn.Sequential(
                nn.Conv1d(planes, planes * 8, kernel_size=1, stride=1, bias=False),
                nn.Identity(),
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv1d(planes, planes * 8, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(planes * 8),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if not self.disable_skip:
            out = out + identity

        out = self.relu(out)
        return out


class ResidualBlock2DAblated(nn.Module):
    """ResidualBlock2D with optional skip connection and BatchNorm ablation."""

    def __init__(
        self,
        planes,
        kernel_size=(11, 5),
        padding=(5, 2),
        disable_skip=False,
        disable_batchnorm=False,
    ):
        super().__init__()
        self.disable_skip = disable_skip

        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.c2 = nn.Conv2d(
            planes, planes * 2, kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
        self.c3 = nn.Conv2d(
            planes * 2, planes * 4, kernel_size=1, stride=1, bias=False
        )

        if disable_batchnorm:
            self.b1 = nn.Identity()
            self.b2 = nn.Identity()
            self.b3 = nn.Identity()
        else:
            self.b1 = nn.BatchNorm2d(planes)
            self.b2 = nn.BatchNorm2d(planes * 2)
            self.b3 = nn.BatchNorm2d(planes * 4)

        # Downsample path (always needed for channel expansion)
        if disable_batchnorm:
            self.downsample = nn.Sequential(
                nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),
                nn.Identity(),
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if not self.disable_skip:
            out = out + identity

        out = self.relu(out)
        return out


class PrismNetAblated(nn.Module):
    """PrismNet with configurable component ablation.

    This model mirrors the original PrismNet architecture but allows
    disabling specific components for ablation studies.
    """

    def __init__(self, config: AblationConfig, mode: str = "pu", base_channel: int = 8):
        super().__init__()
        self.config = config
        self.mode = mode
        self.base_channel = base_channel

        # Mode-specific parameters
        h_p, h_k = 2, 5
        if mode == "pu":
            self.n_features = 5
        elif mode == "seq":
            self.n_features = 4
            h_p, h_k = 1, 3
        elif mode == "str":
            self.n_features = 1
            h_p, h_k = 0, 1
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Dropout rates (0 if disabled)
        self.dropout_rates = (
            (0.0, 0.0, 0.0) if config.disable_dropout else (0.1, 0.5, 0.3)
        )

        # Build layers with ablation settings
        self.conv = Conv2dAblated(
            1,
            base_channel,
            kernel_size=(11, h_k),
            bn=True,
            same_padding=True,
            disable_batchnorm=config.disable_batchnorm,
        )

        if config.disable_se:
            self.se = IdentitySE()
        else:
            from prismnet.model.se import SEBlock
            self.se = SEBlock(base_channel)

        self.res2d = ResidualBlock2DAblated(
            base_channel,
            kernel_size=(11, h_k),
            padding=(5, h_p),
            disable_skip=config.disable_res2d_skip,
            disable_batchnorm=config.disable_batchnorm,
        )

        self.res1d = ResidualBlock1DAblated(
            base_channel * 4,
            disable_skip=config.disable_res1d_skip,
            disable_batchnorm=config.disable_batchnorm,
        )

        self.avgpool = nn.AvgPool2d((1, self.n_features))
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channel * 4 * 8, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # Handle different input modes
        if self.mode == "seq":
            input = input[:, :, :, :4]
        elif self.mode == "str":
            input = input[:, :, :, 4:]

        x = self.conv(input)
        x = F.dropout(x, self.dropout_rates[0], training=self.training)

        z = self.se(x)
        x = self.res2d(x * z)
        x = F.dropout(x, self.dropout_rates[1], training=self.training)

        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])

        x = self.res1d(x)
        x = F.dropout(x, self.dropout_rates[2], training=self.training)

        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)

        return x


def create_ablated_model(
    config: AblationConfig,
    mode: str = "pu",
    base_channel: int = 8,
) -> nn.Module:
    """Factory function to create an ablated PrismNet model.

    Args:
        config: Ablation configuration specifying which components to disable
        mode: Input mode ("pu", "seq", or "str")
        base_channel: Base channel width (8 for standard, 64 for large)

    Returns:
        PrismNetAblated model with specified ablations applied
    """
    return PrismNetAblated(config, mode=mode, base_channel=base_channel)
