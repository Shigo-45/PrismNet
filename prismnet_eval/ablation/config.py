"""Ablation experiment configurations."""

from dataclasses import dataclass, field


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


def get_config_by_name(name: str) -> AblationConfig:
    """Get an ablation config by name from predefined configs."""
    all_configs = LEAVE_ONE_OUT_CONFIGS + CUMULATIVE_CONFIGS
    for config in all_configs:
        if config.name == name:
            return config
    raise ValueError(f"Unknown ablation config: {name}")
