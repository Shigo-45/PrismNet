"""Model randomization tests for saliency sanity checks.

This module implements tests from the paper "Sanity Checks for Saliency Maps"
(Adebayo et al., NeurIPS 2018) to verify that saliency maps are model-dependent.
"""

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from prismnet.model.smoothgrad import GuidedBackpropSmoothGrad


def compute_similarity_metrics(
    saliency1: torch.Tensor, saliency2: torch.Tensor
) -> Dict[str, float]:
    """Compute similarity metrics between two saliency maps.

    Args:
        saliency1: First saliency map tensor (batch, 1, seq_len, features)
        saliency2: Second saliency map tensor (batch, 1, seq_len, features)

    Returns:
        Dictionary with SSIM, Spearman correlation, L2 distance, and sample counts
    """
    # Convert to numpy and flatten
    sal1_np = saliency1.detach().cpu().numpy()
    sal2_np = saliency2.detach().cpu().numpy()

    # Compute metrics across the batch
    ssim_scores = []
    spearman_scores = []
    l2_distances = []

    total_samples = sal1_np.shape[0]

    for i in range(total_samples):
        # Flatten spatial dimensions for each sample
        s1_flat = sal1_np[i].flatten()
        s2_flat = sal2_np[i].flatten()

        # SSIM requires 2D arrays, use first channel if multi-channel
        s1_2d = sal1_np[i, 0, :, :]
        s2_2d = sal2_np[i, 0, :, :]

        # Compute SSIM with adaptive window size
        data_range = max(s1_2d.max() - s1_2d.min(), s2_2d.max() - s2_2d.min())
        if data_range > 0:
            # Determine appropriate window size based on image dimensions
            min_dim = min(s1_2d.shape)
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            if win_size >= 3:  # SSIM requires at least 3x3 window
                ssim_val = ssim(s1_2d, s2_2d, data_range=data_range, win_size=win_size)
                ssim_scores.append(ssim_val)

        # Compute Spearman correlation
        if len(np.unique(s1_flat)) > 1 and len(np.unique(s2_flat)) > 1:
            corr, _ = spearmanr(s1_flat, s2_flat)
            if not np.isnan(corr):
                spearman_scores.append(corr)

        # Compute L2 distance
        l2_dist = np.linalg.norm(s1_flat - s2_flat)
        l2_distances.append(l2_dist)

    # Calculate dropout rates
    n_valid_ssim = len(ssim_scores)
    n_valid_spearman = len(spearman_scores)
    ssim_dropout_rate = (total_samples - n_valid_ssim) / total_samples
    spearman_dropout_rate = (total_samples - n_valid_spearman) / total_samples

    # Warn if high dropout
    if ssim_dropout_rate > 0.1:
        print(f"Warning: {ssim_dropout_rate*100:.1f}% of samples dropped for SSIM calculation")
    if spearman_dropout_rate > 0.1:
        print(f"Warning: {spearman_dropout_rate*100:.1f}% of samples dropped for Spearman calculation")

    return {
        "ssim_mean": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        "ssim_std": float(np.std(ssim_scores)) if ssim_scores else 0.0,
        "ssim_n_valid": n_valid_ssim,
        "ssim_n_total": total_samples,
        "spearman_mean": float(np.mean(spearman_scores)) if spearman_scores else 0.0,
        "spearman_std": float(np.std(spearman_scores)) if spearman_scores else 0.0,
        "spearman_n_valid": n_valid_spearman,
        "spearman_n_total": total_samples,
        "l2_mean": float(np.mean(l2_distances)),
        "l2_std": float(np.std(l2_distances)),
    }


def _randomize_model_weights(model: nn.Module) -> nn.Module:
    """Randomize all weights in a model by reinitializing.

    Uses the same initialization scheme as PrismNet's _initialize_weights():
    - Conv2d/Conv1d: Kaiming normal (fan_out, relu)
    - BatchNorm: weight=1, bias=0
    - Linear: normal(0, 0.01)

    This ensures the random baseline uses the same weight distribution
    as the original training initialization.

    Args:
        model: PyTorch model to randomize

    Returns:
        Model with randomized weights
    """
    # Create a deep copy to avoid modifying original
    model_copy = copy.deepcopy(model)

    # Reinitialize all parameters using PrismNet's initialization scheme
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    model_copy.apply(init_weights)
    return model_copy


def _compute_saliency_for_samples(
    model: nn.Module,
    data_loader: DataLoader,
    n_samples: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Compute saliency maps for n_samples from the data loader.

    Args:
        model: Model to compute saliency for
        data_loader: DataLoader providing samples
        n_samples: Number of samples to process
        device: Device to run on

    Returns:
        Tensor of saliency maps (n_samples, 1, seq_len, features)
    """
    model = model.to(device)
    model.eval()

    # Create saliency computer
    saliency_computer = GuidedBackpropSmoothGrad(
        model, device=device, only_seq=False, nsamples=20, magnitude=2
    )

    saliency_maps = []
    samples_processed = 0

    for batch_x, _ in data_loader:
        if samples_processed >= n_samples:
            break

        batch_x = batch_x.to(device)
        batch_size = min(batch_x.size(0), n_samples - samples_processed)

        # Compute saliency for this batch
        batch_saliency = saliency_computer.get_batch_gradients(batch_x[:batch_size])
        # Detach and move to CPU to free computation graph and GPU memory
        saliency_maps.append(batch_saliency.detach().cpu())

        samples_processed += batch_size

    return torch.cat(saliency_maps, dim=0)


def full_randomization_test(
    trained_model: nn.Module,
    test_loader: DataLoader,
    n_random: int = 5,
    n_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """Compare saliency from trained vs fully random models.

    This test verifies that saliency maps depend on trained model weights.
    If saliency is meaningful, it should differ significantly between trained
    and random models.

    Args:
        trained_model: Trained model to test
        test_loader: DataLoader for test samples
        n_random: Number of random model initializations
        n_samples: Number of samples to test on

    Returns:
        Dictionary with similarity statistics and per-model comparisons
    """
    # Input validation
    if n_random <= 0:
        raise ValueError(f"n_random must be positive, got {n_random}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if trained_model.training:
        raise ValueError("trained_model must be in eval mode (call model.eval())")
    if len(test_loader.dataset) == 0:
        raise ValueError("test_loader has empty dataset")

    print(f"Computing saliency for trained model on {n_samples} samples...")
    trained_saliency = _compute_saliency_for_samples(
        trained_model, test_loader, n_samples, device
    )

    # Compare trained model to random initializations
    trained_vs_random_metrics = []

    for i in tqdm(range(n_random), desc="Testing random initializations"):
        # Create random model
        random_model = _randomize_model_weights(trained_model)

        # Compute saliency
        random_saliency = _compute_saliency_for_samples(
            random_model, test_loader, n_samples, device
        )

        # Compare
        metrics = compute_similarity_metrics(trained_saliency, random_saliency)
        trained_vs_random_metrics.append(metrics)

    # Also compare random models to each other (baseline)
    random_vs_random_metrics = []
    print("Computing baseline: random vs random comparisons...")

    for i in tqdm(range(min(n_random, 3)), desc="Random-vs-random baseline"):
        random1 = _randomize_model_weights(trained_model)
        random2 = _randomize_model_weights(trained_model)

        sal1 = _compute_saliency_for_samples(random1, test_loader, n_samples, device)
        sal2 = _compute_saliency_for_samples(random2, test_loader, n_samples, device)

        metrics = compute_similarity_metrics(sal1, sal2)
        random_vs_random_metrics.append(metrics)

    # Aggregate statistics
    def aggregate_metrics(metric_list: List[Dict]) -> Dict:
        """Average metrics across multiple comparisons."""
        if not metric_list:
            raise ValueError("Cannot aggregate metrics from empty list")
        keys = metric_list[0].keys()
        return {
            key: {
                "mean": float(np.mean([m[key] for m in metric_list])),
                "std": float(np.std([m[key] for m in metric_list])),
            }
            for key in keys
        }

    return {
        "trained_vs_random": aggregate_metrics(trained_vs_random_metrics),
        "random_vs_random": aggregate_metrics(random_vs_random_metrics),
        "n_samples": n_samples,
        "n_random_models": n_random,
        "interpretation": (
            "Lower SSIM and Spearman for trained-vs-random compared to "
            "random-vs-random indicates saliency depends on learned weights."
        ),
    }


def cascading_randomization_test(
    trained_model: nn.Module,
    test_loader: DataLoader,
    n_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """Progressively randomize layers from top (output) to bottom (input).

    This test identifies which layers the saliency computation depends on.
    Randomizing early layers (close to output) should affect saliency more
    than later layers if the method is model-dependent.

    Args:
        trained_model: Trained model to test
        test_loader: DataLoader for test samples
        n_samples: Number of samples to test on

    Returns:
        Dictionary with per-layer similarity statistics
    """
    # Input validation
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if trained_model.training:
        raise ValueError("trained_model must be in eval mode (call model.eval())")
    if len(test_loader.dataset) == 0:
        raise ValueError("test_loader has empty dataset")

    print(f"Computing saliency for trained model on {n_samples} samples...")
    trained_saliency = _compute_saliency_for_samples(
        trained_model, test_loader, n_samples, device
    )

    # Define layer order from output to input for PrismNet
    # Based on forward pass: conv -> se -> res2d -> avgpool -> res1d -> gpool -> fc
    # Cascading order (output to input): fc -> gpool -> res1d -> avgpool -> res2d -> se -> conv
    layer_names = ["fc", "res1d", "res2d", "se", "conv"]

    results = {}

    for i, layer_name in enumerate(layer_names):
        print(f"\nRandomizing layers up to and including: {layer_name}")

        # Create model copy
        model_copy = copy.deepcopy(trained_model)

        # Randomize layers from fc down to current layer
        for j in range(i + 1):
            target_layer = layer_names[j]
            if hasattr(model_copy, target_layer):
                layer = getattr(model_copy, target_layer)

                # Reinitialize this layer
                def init_weights(m):
                    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu"
                        )
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)

                layer.apply(init_weights)

        # Compute saliency with partially randomized model
        partial_saliency = _compute_saliency_for_samples(
            model_copy, test_loader, n_samples, device
        )

        # Compare to trained model
        metrics = compute_similarity_metrics(trained_saliency, partial_saliency)

        results[layer_name] = metrics

    return {
        "layer_results": results,
        "layer_order": layer_names,
        "n_samples": n_samples,
        "interpretation": (
            "Saliency should diverge (low SSIM/Spearman) when critical layers "
            "are randomized. The layer where divergence occurs indicates where "
            "saliency computation depends on learned features."
        ),
    }
