"""MLflow experiment tracking integration."""

import json
from pathlib import Path
from typing import Any, Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def check_mlflow():
    """Check if MLflow is available."""
    if not MLFLOW_AVAILABLE:
        raise ImportError(
            "MLflow not installed. Install with: pip install mlflow "
            "or: uv pip install mlflow"
        )


def setup_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
) -> str:
    """Create or get an MLflow experiment.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI (defaults to local ./mlruns)

    Returns:
        Experiment ID
    """
    check_mlflow()

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_ablation_run(
    config_name: str,
    config_dict: dict,
    metrics: dict,
    training_config: dict,
    dataset: str,
    artifacts: Optional[list[str]] = None,
    run_name: Optional[str] = None,
):
    """Log an ablation experiment run to MLflow.

    Args:
        config_name: Name of the ablation configuration
        config_dict: Ablation config as dictionary
        metrics: Dictionary of metrics (auc, prc, acc, etc.)
        training_config: Training hyperparameters
        dataset: Dataset name
        artifacts: List of artifact file paths to log
        run_name: Optional name for the run
    """
    check_mlflow()

    with mlflow.start_run(run_name=run_name or f"{dataset}_{config_name}"):
        # Log parameters
        mlflow.log_param("ablation_config", config_name)
        mlflow.log_param("dataset", dataset)

        for key, value in config_dict.items():
            mlflow.log_param(f"ablation.{key}", value)

        for key, value in training_config.items():
            mlflow.log_param(f"training.{key}", value)

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # Log artifacts
        if artifacts:
            for artifact_path in artifacts:
                if Path(artifact_path).exists():
                    mlflow.log_artifact(artifact_path)


def log_training_curve(
    epoch: int,
    train_loss: float,
    train_auc: float,
    val_loss: float,
    val_auc: float,
    lr: float,
):
    """Log training metrics for a single epoch.

    Should be called within an active MLflow run context.
    """
    check_mlflow()

    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_auc", train_auc, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_auc", val_auc, step=epoch)
    mlflow.log_metric("learning_rate", lr, step=epoch)


class AblationTracker:
    """Context manager for tracking ablation experiments with MLflow."""

    def __init__(
        self,
        experiment_name: str = "prismnet-ablation",
        tracking_uri: Optional[str] = None,
        enabled: bool = True,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.experiment_id = None

    def __enter__(self):
        if self.enabled:
            self.experiment_id = setup_experiment(
                self.experiment_name, self.tracking_uri
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def start_run(
        self,
        config_name: str,
        dataset: str,
        config_dict: dict,
        training_config: dict,
    ):
        """Start a new run for an ablation experiment."""
        if not self.enabled:
            return NullContext()

        run = mlflow.start_run(run_name=f"{dataset}_{config_name}")

        # Log parameters
        mlflow.log_param("ablation_config", config_name)
        mlflow.log_param("dataset", dataset)

        for key, value in config_dict.items():
            mlflow.log_param(f"ablation.{key}", value)

        for key, value in training_config.items():
            mlflow.log_param(f"training.{key}", value)

        return run

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_auc: float,
        val_loss: float,
        val_auc: float,
        lr: float,
    ):
        """Log metrics for a training epoch."""
        if not self.enabled:
            return

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_auc", train_auc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_auc", val_auc, step=epoch)
        mlflow.log_metric("learning_rate", lr, step=epoch)

    def log_final_metrics(self, metrics: dict):
        """Log final metrics for the run."""
        if not self.enabled:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"best_{key}", value)

    def log_artifact(self, path: str):
        """Log an artifact file."""
        if not self.enabled:
            return

        if Path(path).exists():
            mlflow.log_artifact(path)

    def end_run(self):
        """End the current run."""
        if self.enabled:
            mlflow.end_run()


class NullContext:
    """Null context manager for when MLflow is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
