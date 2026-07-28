"""Visualization functions for saliency sanity checks."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_saliency_comparison(
    trained_saliency: torch.Tensor,
    random_saliency: torch.Tensor,
    sample_indices: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Create side-by-side comparison of trained vs random model saliency.

    Args:
        trained_saliency: Saliency from trained model (n_samples, 1, seq_len, features)
        random_saliency: Saliency from random model (n_samples, 1, seq_len, features)
        sample_indices: Which samples to visualize (default: first 5)
        output_path: Where to save the plot

    Returns:
        Path to saved plot
    """
    if sample_indices is None:
        sample_indices = list(range(min(5, trained_saliency.shape[0])))

    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(sample_indices):
        # Get saliency maps for this sample
        trained_sal = trained_saliency[idx, 0].cpu().numpy()
        random_sal = random_saliency[idx, 0].cpu().numpy()

        # Plot trained model saliency
        im1 = axes[i, 0].imshow(
            trained_sal.T, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        axes[i, 0].set_title(f"Sample {idx}: Trained Model")
        axes[i, 0].set_xlabel("Sequence Position")
        axes[i, 0].set_ylabel("Feature")
        plt.colorbar(im1, ax=axes[i, 0])

        # Plot random model saliency
        im2 = axes[i, 1].imshow(
            random_sal.T, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        axes[i, 1].set_title(f"Sample {idx}: Random Model")
        axes[i, 1].set_xlabel("Sequence Position")
        axes[i, 1].set_ylabel("Feature")
        plt.colorbar(im2, ax=axes[i, 1])

    plt.tight_layout()

    if output_path is None:
        output_path = Path("saliency_comparison.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_similarity_distributions(
    trained_vs_random_metrics: List[Dict[str, float]],
    random_vs_random_metrics: List[Dict[str, float]],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot distributions of similarity metrics.

    Args:
        trained_vs_random_metrics: List of metric dicts from trained-vs-random
        random_vs_random_metrics: List of metric dicts from random-vs-random
        output_path: Where to save the plot

    Returns:
        Path to saved plot
    """
    # Extract metrics
    metrics_to_plot = ["ssim_mean", "spearman_mean"]
    metric_labels = {"ssim_mean": "SSIM", "spearman_mean": "Spearman Correlation"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric in zip(axes, metrics_to_plot):
        # Extract values
        trained_vs_random = [m[metric] for m in trained_vs_random_metrics]
        random_vs_random = [m[metric] for m in random_vs_random_metrics]

        # Create violin plots
        positions = [1, 2]
        ax.violinplot(
            [trained_vs_random, random_vs_random],
            positions=positions,
            showmeans=True,
            showmedians=True,
        )

        # Customize
        ax.set_xticks(positions)
        ax.set_xticklabels(["Trained vs\nRandom", "Random vs\nRandom"])
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f"{metric_labels[metric]} Distribution")
        ax.grid(axis="y", alpha=0.3)

        # Add mean values as text
        mean_tvr = np.mean(trained_vs_random)
        mean_rvr = np.mean(random_vs_random)
        ax.text(
            1,
            mean_tvr,
            f"{mean_tvr:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            2,
            mean_rvr,
            f"{mean_rvr:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.suptitle(
        "Saliency Similarity: Trained vs Random Models\n"
        "(Lower similarity indicates model-dependent saliency)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    if output_path is None:
        output_path = Path("similarity_distributions.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_cascading_results(
    layer_results: Dict[str, Dict[str, float]],
    layer_order: List[str],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot cascading randomization test results.

    Args:
        layer_results: Dictionary mapping layer names to similarity metrics
        layer_order: Order of layers from output to input
        output_path: Where to save the plot

    Returns:
        Path to saved plot
    """
    # Extract metrics for each layer
    ssim_values = [layer_results[layer]["ssim_mean"] for layer in layer_order]
    spearman_values = [layer_results[layer]["spearman_mean"] for layer in layer_order]

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SSIM plot
    axes[0].plot(
        range(len(layer_order)), ssim_values, marker="o", linewidth=2, markersize=8
    )
    axes[0].set_xticks(range(len(layer_order)))
    axes[0].set_xticklabels(layer_order, rotation=45, ha="right")
    axes[0].set_ylabel("SSIM")
    axes[0].set_xlabel("Layers Randomized (Output → Input)")
    axes[0].set_title("SSIM vs Layer Randomization")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Threshold")

    # Add value labels
    for i, val in enumerate(ssim_values):
        axes[0].text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    axes[0].legend()

    # Spearman plot
    axes[1].plot(
        range(len(layer_order)),
        spearman_values,
        marker="o",
        linewidth=2,
        markersize=8,
        color="orange",
    )
    axes[1].set_xticks(range(len(layer_order)))
    axes[1].set_xticklabels(layer_order, rotation=45, ha="right")
    axes[1].set_ylabel("Spearman Correlation")
    axes[1].set_xlabel("Layers Randomized (Output → Input)")
    axes[1].set_title("Spearman Correlation vs Layer Randomization")
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Threshold")

    # Add value labels
    for i, val in enumerate(spearman_values):
        axes[1].text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].legend()

    plt.suptitle(
        "Cascading Randomization Test: Saliency Divergence by Layer\n"
        "(Lower values indicate saliency depends on that layer)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    if output_path is None:
        output_path = Path("cascading_results.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_summary_report(
    full_test_results: Dict,
    cascading_results: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """Create a comprehensive summary visualization.

    Args:
        full_test_results: Results from full_randomization_test()
        cascading_results: Results from cascading_randomization_test()
        output_path: Where to save the plot

    Returns:
        Path to saved plot
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Extract data
    tvr = full_test_results["trained_vs_random"]
    rvr = full_test_results["random_vs_random"]
    layer_results = cascading_results["layer_results"]
    layer_order = cascading_results["layer_order"]

    # 1. SSIM comparison bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ["Trained vs\nRandom", "Random vs\nRandom"]
    ssim_means = [tvr["ssim_mean"]["mean"], rvr["ssim_mean"]["mean"]]
    ssim_stds = [tvr["ssim_mean"]["std"], rvr["ssim_mean"]["std"]]

    bars = ax1.bar(categories, ssim_means, yerr=ssim_stds, capsize=5, alpha=0.7)
    bars[0].set_color("steelblue")
    bars[1].set_color("orange")
    ax1.set_ylabel("SSIM")
    ax1.set_title("Full Randomization: SSIM Comparison")
    ax1.grid(axis="y", alpha=0.3)

    for i, (mean, std) in enumerate(zip(ssim_means, ssim_stds)):
        ax1.text(i, mean, f"{mean:.3f}\n±{std:.3f}", ha="center", va="bottom")

    # 2. Spearman comparison bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    spearman_means = [tvr["spearman_mean"]["mean"], rvr["spearman_mean"]["mean"]]
    spearman_stds = [tvr["spearman_mean"]["std"], rvr["spearman_mean"]["std"]]

    bars = ax2.bar(categories, spearman_means, yerr=spearman_stds, capsize=5, alpha=0.7)
    bars[0].set_color("steelblue")
    bars[1].set_color("orange")
    ax2.set_ylabel("Spearman Correlation")
    ax2.set_title("Full Randomization: Spearman Comparison")
    ax2.grid(axis="y", alpha=0.3)

    for i, (mean, std) in enumerate(zip(spearman_means, spearman_stds)):
        ax2.text(i, mean, f"{mean:.3f}\n±{std:.3f}", ha="center", va="bottom")

    # 3. Cascading SSIM
    ax3 = fig.add_subplot(gs[1, :])
    ssim_cascade = [layer_results[layer]["ssim_mean"] for layer in layer_order]
    ax3.plot(range(len(layer_order)), ssim_cascade, marker="o", linewidth=2, markersize=10, color="steelblue")
    ax3.set_xticks(range(len(layer_order)))
    ax3.set_xticklabels(layer_order, fontsize=11)
    ax3.set_ylabel("SSIM", fontsize=11)
    ax3.set_xlabel("Layers Randomized (Output → Input)", fontsize=11)
    ax3.set_title("Cascading Randomization: SSIM by Layer", fontsize=12)
    ax3.grid(alpha=0.3)
    ax3.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

    for i, val in enumerate(ssim_cascade):
        ax3.text(i, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)

    # 4. Cascading Spearman
    ax4 = fig.add_subplot(gs[2, :])
    spearman_cascade = [layer_results[layer]["spearman_mean"] for layer in layer_order]
    ax4.plot(range(len(layer_order)), spearman_cascade, marker="o", linewidth=2, markersize=10, color="orange")
    ax4.set_xticks(range(len(layer_order)))
    ax4.set_xticklabels(layer_order, fontsize=11)
    ax4.set_ylabel("Spearman Correlation", fontsize=11)
    ax4.set_xlabel("Layers Randomized (Output → Input)", fontsize=11)
    ax4.set_title("Cascading Randomization: Spearman by Layer", fontsize=12)
    ax4.grid(alpha=0.3)
    ax4.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

    for i, val in enumerate(spearman_cascade):
        ax4.text(i, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)

    # Overall title
    fig.suptitle(
        "Saliency Sanity Check Summary Report\n"
        f"Full test: {full_test_results['n_samples']} samples, "
        f"{full_test_results['n_random_models']} random models",
        fontsize=14,
        fontweight="bold",
    )

    if output_path is None:
        output_path = Path("saliency_summary_report.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path
