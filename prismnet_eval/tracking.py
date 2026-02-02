"""MLflow experiment tracking integration."""

import mlflow
from pathlib import Path
from typing import Dict, List


def setup_experiment(name: str) -> str:
    """Create or get MLflow experiment, return experiment_id."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def log_splitting_run(method: str, leakage_stats: Dict, model_metrics: Dict = None):
    """Log homology splitting experiment results."""
    with mlflow.start_run():
        mlflow.log_param("splitting_method", method)
        mlflow.log_metrics(leakage_stats)
        if model_metrics:
            mlflow.log_metrics(model_metrics)
