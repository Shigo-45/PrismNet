"""Ablation experiment runner."""

import fcntl
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from prismnet.loader import SeqicSHAPE
from prismnet.model.utils import GradualWarmupScheduler

from .config import AblationConfig, BaselineConfig
from .variants import create_ablated_model


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    lr: float = 0.001
    batch_size: int = 64
    nepochs: int = 200
    pos_weight: int = 2
    weight_decay: float = 1e-6
    early_stopping: int = 20
    seed: int = 1024
    mode: str = "pu"


@dataclass
class AblationResult:
    """Results from an ablation experiment."""

    config_name: str
    dataset: str
    best_auc: float
    best_prc: float
    best_acc: float
    best_epoch: int
    train_time_seconds: float
    model_path: str

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "dataset": self.dataset,
            "best_auc": self.best_auc,
            "best_prc": self.best_prc,
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "train_time_seconds": self.train_time_seconds,
            "model_path": self.model_path,
        }


def fix_seed(seed: int):
    """Seed all random number generators for reproducibility."""
    torch.set_num_threads(1)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, device, train_loader, criterion, optimizer):
    """Run one training epoch."""
    from prismnet.utils import metrics

    model.train()
    met = metrics.MLMetrics(objective="binary")

    for batch_idx, (x0, y0) in enumerate(train_loader):
        x = x0.float().to(device)
        y = y0.to(device).float()

        # Skip batches with all same labels
        if y0.sum() == 0 or y0.sum() == len(y0):
            continue

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        prob = torch.sigmoid(output)

        y_np = y.cpu().long().detach().numpy()
        p_np = prob.cpu().detach().numpy()
        met.update(y_np, p_np, [loss.item()])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return met


def validate_epoch(model, device, test_loader, criterion):
    """Run validation."""
    from prismnet.utils import metrics

    model.eval()
    y_all = []
    p_all = []
    l_all = []

    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x = x0.float().to(device)
            y = y0.to(device).float()

            output = model(x)
            loss = criterion(output, y)
            prob = torch.sigmoid(output)

            y_np = y.cpu().long().numpy()
            p_np = prob.cpu().numpy()

            y_all.append(y_np)
            p_all.append(p_np)
            l_all.append(loss.item())

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)

    met = metrics.MLMetrics(objective="binary")
    met.update(y_all, p_all, [l_all.mean()])

    return met, y_all, p_all


def run_ablation_experiment(
    ablation_config,  # AblationConfig or BaselineConfig
    data_path: str,
    output_dir: str,
    training_config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> AblationResult:
    """Run a single ablation experiment.

    Args:
        ablation_config: AblationConfig or BaselineConfig specifying the model
        data_path: Path to HDF5 dataset file
        output_dir: Directory to save results and models
        training_config: Training hyperparameters (uses defaults if None)
        device: PyTorch device (auto-detects CUDA if None)
        verbose: Whether to print progress

    Returns:
        AblationResult with metrics and model path
    """
    import multiprocessing as mp

    if training_config is None:
        training_config = TrainingConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup
    fix_seed(training_config.seed)
    dataset_name = Path(data_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{dataset_name}_{ablation_config.name}.pth"

    # Detect if we're in a daemonic process (parallel execution)
    # Daemonic processes can't spawn children, so set num_workers=0
    current_process = mp.current_process()
    is_daemon = current_process.daemon if hasattr(current_process, 'daemon') else False
    num_workers = 0 if is_daemon else 2

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        SeqicSHAPE(data_path),
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        SeqicSHAPE(data_path, is_test=True),
        batch_size=training_config.batch_size * 8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if verbose:
        print(f"Dataset: {dataset_name}")
        print(f"  Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
        print(f"Config: {ablation_config.name}")
        print(f"  {ablation_config.to_dict()}")

    # Model - check if it's a baseline or ablation config
    if isinstance(ablation_config, BaselineConfig):
        model = ablation_config.model_fn()
    else:
        model = create_ablated_model(
            ablation_config, mode=training_config.mode, base_channel=8
        )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(training_config.pos_weight, dtype=torch.float32)
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.lr,
        betas=(0.9, 0.999),
        weight_decay=training_config.weight_decay,
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=8,
        total_epoch=float(training_config.nepochs),
        after_scheduler=None,
    )

    # Training loop
    best_auc = 0
    best_prc = 0
    best_acc = 0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, training_config.nepochs + 1):
        t_met = train_epoch(model, device, train_loader, criterion, optimizer)
        v_met, _, _ = validate_epoch(model, device, test_loader, criterion)
        scheduler.step()

        if v_met.auc > best_auc:
            best_auc = v_met.auc
            best_prc = v_met.prc
            best_acc = v_met.acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        if epoch - best_epoch > training_config.early_stopping:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

        if verbose and epoch % 10 == 0:
            lr = scheduler.get_lr()[0]
            print(
                f"  Epoch {epoch}: train_auc={t_met.auc:.4f}, "
                f"val_auc={v_met.auc:.4f}, best={best_auc:.4f}, lr={lr:.6f}"
            )

    train_time = time.time() - start_time

    if verbose:
        print(f"  Final: AUC={best_auc:.4f}, PRC={best_prc:.4f}, ACC={best_acc:.4f}")
        print(f"  Best epoch: {best_epoch}, Time: {train_time:.1f}s")

    return AblationResult(
        config_name=ablation_config.name,
        dataset=dataset_name,
        best_auc=best_auc,
        best_prc=best_prc,
        best_acc=best_acc,
        best_epoch=best_epoch,
        train_time_seconds=train_time,
        model_path=str(model_path),
    )


def run_ablation_suite(
    configs: list[AblationConfig],
    data_paths: list[str],
    output_dir: str,
    training_config: Optional[TrainingConfig] = None,
    verbose: bool = True,
) -> list[AblationResult]:
    """Run ablation experiments for multiple configs and datasets.

    Args:
        configs: List of ablation configurations
        data_paths: List of dataset paths
        output_dir: Directory to save all results
        training_config: Training hyperparameters
        verbose: Print progress

    Returns:
        List of all AblationResult objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    total = len(configs) * len(data_paths)
    current = 0

    for data_path in data_paths:
        for config in configs:
            current += 1
            if verbose:
                print(f"\n[{current}/{total}] {Path(data_path).stem} - {config.name}")

            result = run_ablation_experiment(
                config,
                data_path,
                output_dir,
                training_config=training_config,
                verbose=verbose,
            )
            results.append(result)

            # Save intermediate results with file locking
            results_path = output_dir / "ablation_results.json"

            # Use file locking to prevent concurrent writes
            with open(results_path, "a+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    # Read existing results
                    f.seek(0)
                    content = f.read()
                    if content.strip():
                        existing_results = json.loads(content)
                    else:
                        existing_results = []

                    # Add new result
                    existing_results.append(result.to_dict())

                    # Write back all results
                    f.seek(0)
                    f.truncate()
                    json.dump(existing_results, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return results
