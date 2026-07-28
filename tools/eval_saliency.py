#!/usr/bin/env python
"""CLI tool for running saliency sanity checks on PrismNet models.

This tool implements model randomization tests to verify that saliency maps
depend on learned model weights rather than being artifacts of the gradient
computation method.

Usage:
    python tools/eval_saliency.py \\
        --model-path exp/prismnet/out/ckpt/TIA1_Hela_best.pth \\
        --dataset data/clip_data/TIA1_Hela.h5 \\
        --n-samples 100 \\
        --output-dir evaluation/saliency/TIA1_Hela

References:
    Adebayo et al. (2018). "Sanity Checks for Saliency Maps." NeurIPS.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prismnet.loader import SeqicSHAPE
from prismnet.model.PrismNet import PrismNet, PrismNet_large
from prismnet_eval.saliency import (
    cascading_randomization_test,
    full_randomization_test,
)
from prismnet_eval.saliency.visualization import (
    plot_cascading_results,
    plot_summary_report,
)


def load_model(model_path: Path, mode: str = "pu", large: bool = False) -> torch.nn.Module:
    """Load a trained PrismNet model.

    Args:
        model_path: Path to model checkpoint
        mode: Model mode (pu, seq, or str)
        large: Whether to use PrismNet_large architecture

    Returns:
        Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    if large:
        model = PrismNet_large(mode=mode)
    else:
        model = PrismNet(mode=mode)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def create_data_loader(
    dataset_path: Path, batch_size: int = 32, mode: str = "pu"
) -> DataLoader:
    """Create a DataLoader for the test set.

    Args:
        dataset_path: Path to HDF5 dataset
        batch_size: Batch size for loading
        mode: Data mode (pu, seq, or str)

    Returns:
        DataLoader for test set
    """
    # SeqicSHAPE uses is_test parameter, not train
    # use_structure determines if structure features are included
    use_structure = mode in ["pu", "str"]
    dataset = SeqicSHAPE(dataset_path, is_test=True, use_structure=use_structure)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return loader


def main():
    parser = argparse.ArgumentParser(
        description="Run saliency sanity checks on PrismNet models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to HDF5 dataset file for testing",
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/saliency"),
        help="Directory to save results (default: evaluation/saliency)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to test (default: 100)",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=5,
        help="Number of random model initializations for full test (default: 5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pu",
        choices=["pu", "seq", "str"],
        help="Model mode: pu (protein+structure), seq (sequence only), str (structure only)",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Use PrismNet_large architecture (default: False)",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        default="both",
        choices=["full", "cascading", "both"],
        help="Type of test to run (default: both)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, mode=args.mode, large=args.large)
    print(f"Model loaded successfully (mode={args.mode}, large={args.large})")

    # Create data loader
    print(f"Loading dataset from {args.dataset}...")
    test_loader = create_data_loader(args.dataset, args.batch_size, args.mode)
    print(f"Dataset loaded: {len(test_loader.dataset)} samples in test set")

    # Verify we have enough samples
    if len(test_loader.dataset) < args.n_samples:
        print(
            f"Warning: Requested {args.n_samples} samples but dataset only has "
            f"{len(test_loader.dataset)}. Using all available samples."
        )
        args.n_samples = len(test_loader.dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Run tests
    results = {}

    if args.test_type in ["full", "both"]:
        print("\n" + "=" * 70)
        print("RUNNING FULL RANDOMIZATION TEST")
        print("=" * 70)
        print(
            f"Testing {args.n_samples} samples against {args.n_random} random initializations..."
        )

        full_results = full_randomization_test(
            model, test_loader, n_random=args.n_random, n_samples=args.n_samples, device=device
        )

        results["full_randomization"] = full_results

        # Print summary
        print("\nFull Randomization Test Results:")
        print("-" * 70)
        tvr = full_results["trained_vs_random"]
        rvr = full_results["random_vs_random"]

        print("Trained vs Random:")
        print(f"  SSIM:     {tvr['ssim_mean']['mean']:.4f} ± {tvr['ssim_mean']['std']:.4f}")
        print(f"  Spearman: {tvr['spearman_mean']['mean']:.4f} ± {tvr['spearman_mean']['std']:.4f}")

        print("\nRandom vs Random (baseline):")
        print(f"  SSIM:     {rvr['ssim_mean']['mean']:.4f} ± {rvr['ssim_mean']['std']:.4f}")
        print(f"  Spearman: {rvr['spearman_mean']['mean']:.4f} ± {rvr['spearman_mean']['std']:.4f}")

        print(f"\n{full_results['interpretation']}")

        # Save results
        results_path = args.output_dir / "full_randomization_results.json"
        with open(results_path, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    if args.test_type in ["cascading", "both"]:
        print("\n" + "=" * 70)
        print("RUNNING CASCADING RANDOMIZATION TEST")
        print("=" * 70)
        print(f"Testing {args.n_samples} samples with progressive layer randomization...")

        cascading_results = cascading_randomization_test(
            model, test_loader, n_samples=args.n_samples, device=device
        )

        results["cascading_randomization"] = cascading_results

        # Print summary
        print("\nCascading Randomization Test Results:")
        print("-" * 70)
        print(f"Layer order (output → input): {' → '.join(cascading_results['layer_order'])}")
        print("\nSimilarity metrics by layer:")

        for layer in cascading_results["layer_order"]:
            metrics = cascading_results["layer_results"][layer]
            print(f"\n{layer}:")
            print(f"  SSIM:     {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
            print(f"  Spearman: {metrics['spearman_mean']:.4f} ± {metrics['spearman_std']:.4f}")

        print(f"\n{cascading_results['interpretation']}")

        # Save results
        results_path = args.output_dir / "cascading_randomization_results.json"
        with open(results_path, "w") as f:
            json.dump(cascading_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plots = []

    if "full_randomization" in results and "cascading_randomization" in results:
        # Create comprehensive summary report
        plot_path = args.output_dir / "saliency_summary_report.png"
        plot_summary_report(
            results["full_randomization"], results["cascading_randomization"], plot_path
        )
        plots.append(plot_path)
        print(f"Summary report saved to: {plot_path}")

    if "cascading_randomization" in results:
        # Create cascading results plot
        plot_path = args.output_dir / "cascading_results.png"
        plot_cascading_results(
            results["cascading_randomization"]["layer_results"],
            results["cascading_randomization"]["layer_order"],
            plot_path,
        )
        plots.append(plot_path)
        print(f"Cascading results plot saved to: {plot_path}")

    print("\n" + "=" * 70)
    print("SALIENCY SANITY CHECKS COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {args.output_dir}")

    # Save summary
    summary_path = args.output_dir / "summary.json"
    summary = {
        "model_path": str(args.model_path),
        "dataset": str(args.dataset),
        "n_samples": args.n_samples,
        "mode": args.mode,
        "large": args.large,
        "test_type": args.test_type,
        "device": device,
    }

    if "full_randomization" in results:
        summary["full_test_summary"] = {
            "trained_vs_random_ssim": results["full_randomization"]["trained_vs_random"]["ssim_mean"]["mean"],
            "random_vs_random_ssim": results["full_randomization"]["random_vs_random"]["ssim_mean"]["mean"],
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
