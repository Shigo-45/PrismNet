"""Ablation experiment configurations."""

from dataclasses import dataclass, field
from typing import Callable, Optional
import torch.nn as nn


@dataclass
class AblationConfig:
    """Configuration specifying which components to disable in PrismNet.

    Each flag controls whether a specific architectural component is disabled.
    When disabled, the component is replaced with an identity operation or
    removed entirely.

    Attributes:
        name: Human-readable name for this configuration
        disable_se: Disable Squeeze-and-Excitation channel attention
        disable_res2d_skip: Disable skip connection in ResidualBlock2D
        disable_res1d_skip: Disable skip connection in ResidualBlock1D
        disable_dropout: Set all dropout rates to 0.0
        disable_batchnorm: Replace BatchNorm with Identity
    """

    name: str
    disable_se: bool = False
    disable_res2d_skip: bool = False
    disable_res1d_skip: bool = False
    disable_dropout: bool = False
    disable_batchnorm: bool = False

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "name": self.name,
            "disable_se": self.disable_se,
            "disable_res2d_skip": self.disable_res2d_skip,
            "disable_res1d_skip": self.disable_res1d_skip,
            "disable_dropout": self.disable_dropout,
            "disable_batchnorm": self.disable_batchnorm,
        }

    @property
    def is_full_model(self) -> bool:
        """Check if this is the full model (no ablations)."""
        return not any(
            [
                self.disable_se,
                self.disable_res2d_skip,
                self.disable_res1d_skip,
                self.disable_dropout,
                self.disable_batchnorm,
            ]
        )


# Leave-one-out ablation configs: disable one component at a time
LEAVE_ONE_OUT_CONFIGS = [
    AblationConfig("full_model"),
    AblationConfig("no_se", disable_se=True),
    AblationConfig("no_res2d_skip", disable_res2d_skip=True),
    AblationConfig("no_res1d_skip", disable_res1d_skip=True),
    AblationConfig("no_dropout", disable_dropout=True),
    AblationConfig("no_batchnorm", disable_batchnorm=True),
]

# Cumulative ablation configs: progressively add components
# Order: minimal -> +batchnorm -> +dropout -> +res2d_skip -> +res1d_skip -> +SE (full)
CUMULATIVE_CONFIGS = [
    AblationConfig(
        "minimal",
        disable_se=True,
        disable_res2d_skip=True,
        disable_res1d_skip=True,
        disable_dropout=True,
        disable_batchnorm=True,
    ),
    AblationConfig(
        "add_batchnorm",
        disable_se=True,
        disable_res2d_skip=True,
        disable_res1d_skip=True,
        disable_dropout=True,
        disable_batchnorm=False,
    ),
    AblationConfig(
        "add_dropout",
        disable_se=True,
        disable_res2d_skip=True,
        disable_res1d_skip=True,
        disable_dropout=False,
        disable_batchnorm=False,
    ),
    AblationConfig(
        "add_res2d_skip",
        disable_se=True,
        disable_res1d_skip=True,
        disable_dropout=False,
        disable_batchnorm=False,
    ),
    AblationConfig(
        "add_res1d_skip",
        disable_se=True,
        disable_dropout=False,
        disable_batchnorm=False,
    ),
    AblationConfig("full_model"),
]


@dataclass
class BaselineConfig:
    """Configuration for baseline models.

    Baseline models are simple architectures used to test the hypothesis
    that PrismNet's performance comes from data quality rather than
    architectural sophistication.

    Attributes:
        name: Human-readable name for this baseline
        model_fn: Factory function that returns an instance of the model
        description: Brief description of the baseline architecture
    """
    name: str
    model_fn: Callable[[], nn.Module]
    description: str

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "name": self.name,
            "description": self.description,
            "is_baseline": True,
        }


# Import baseline models
from .baselines import PlainCNN2D1D, PlainCNN2D, BiLSTMBaseline, BiGRUBaseline


# Baseline model configurations
BASELINE_CONFIGS = [
    BaselineConfig(
        name="plain_cnn_2d1d",
        model_fn=lambda: PlainCNN2D1D(),
        description="Plain CNN with 2D→1D transition, no residual blocks"
    ),
    BaselineConfig(
        name="plain_cnn_2d",
        model_fn=lambda: PlainCNN2D(),
        description="Pure 2D CNN without 1D transition"
    ),
    BaselineConfig(
        name="bilstm",
        model_fn=lambda: BiLSTMBaseline(),
        description="Bidirectional LSTM baseline"
    ),
    BaselineConfig(
        name="bigru",
        model_fn=lambda: BiGRUBaseline(),
        description="Bidirectional GRU baseline"
    ),
]


def get_config_by_name(name: str) -> AblationConfig:
    """Get an ablation config by name from predefined configs."""
    all_configs = LEAVE_ONE_OUT_CONFIGS + CUMULATIVE_CONFIGS
    for config in all_configs:
        if config.name == name:
            return config
    raise ValueError(f"Unknown ablation config: {name}")


def get_baseline_by_name(name: str) -> BaselineConfig:
    """Get a baseline config by name."""
    for config in BASELINE_CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown baseline config: {name}")


def is_baseline_config(name: str) -> bool:
    """Check if a config name refers to a baseline model."""
    return any(config.name == name for config in BASELINE_CONFIGS)
