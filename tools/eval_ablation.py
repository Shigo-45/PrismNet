#!/usr/bin/env python
"""CLI tool for running PrismNet ablation experiments.

Usage:
    # Run leave-one-out ablations on a single dataset
    python tools/eval_ablation.py --data data/TIA1_Hela.h5 --type leave-one-out

    # Run cumulative ablations on multiple datasets
    python tools/eval_ablation.py --data data/clip_data --type cumulative

    # Run all ablations on all datasets
    python tools/eval_ablation.py --data data/clip_data --type both
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prismnet_eval.ablation import (
    CUMULATIVE_CONFIGS,
    LEAVE_ONE_OUT_CONFIGS,
)
from prismnet_eval.ablation.runner import (
    AblationResult,
    TrainingConfig,
    run_ablation_suite,
)


def find_datasets(data_path: str) -> list[str]:
    """Find all HDF5 dataset files."""
    path = Path(data_path)
    if path.is_file() and path.suffix == ".h5":
        return [str(path)]
    elif path.is_dir():
        h5_files = list(path.glob("*.h5"))
        if not h5_files:
            raise ValueError(f"No .h5 files found in {path}")
        return sorted(str(f) for f in h5_files)
    else:
        raise ValueError(f"Invalid data path: {path}")


def get_configs(ablation_type: str):
    """Get ablation configs based on type."""
    if ablation_type == "leave-one-out":
        return LEAVE_ONE_OUT_CONFIGS
    elif ablation_type == "cumulative":
        return CUMULATIVE_CONFIGS
    elif ablation_type == "both":
        # Combine but avoid duplicate full_model
        configs = list(LEAVE_ONE_OUT_CONFIGS)
        for c in CUMULATIVE_CONFIGS:
            if c.name != "full_model":
                configs.append(c)
        return configs
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")


def print_results_summary(results: list[AblationResult]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)

    # Group by dataset
    datasets = {}
    for r in results:
        if r.dataset not in datasets:
            datasets[r.dataset] = []
        datasets[r.dataset].append(r)

    for dataset, dataset_results in datasets.items():
        print(f"\nDataset: {dataset}")
        print("-" * 60)
        print(f"{'Config':<20} {'AUC':>8} {'PRC':>8} {'ACC':>8} {'Epoch':>6}")
        print("-" * 60)

        # Sort by AUC descending
        sorted_results = sorted(dataset_results, key=lambda x: x.best_auc, reverse=True)
        for r in sorted_results:
            print(
                f"{r.config_name:<20} {r.best_auc:>8.4f} {r.best_prc:>8.4f} "
                f"{r.best_acc:>8.2f} {r.best_epoch:>6}"
            )

        # Show delta from full model
        full_model = next((r for r in sorted_results if r.config_name == "full_model"), None)
        if full_model:
            print("\nDelta from full model:")
            for r in sorted_results:
                if r.config_name != "full_model":
                    delta = r.best_auc - full_model.best_auc
                    sign = "+" if delta > 0 else ""
                    print(f"  {r.config_name:<20} {sign}{delta:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run PrismNet ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to HDF5 file or directory containing HDF5 files",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="leave-one-out",
        choices=["leave-one-out", "cumulative", "both"],
        help="Type of ablation study to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/ablation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Random seed",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pu",
        choices=["pu", "seq", "str"],
        help="Input mode",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    args = parser.parse_args()

    # Find datasets
    try:
        data_paths = find_datasets(args.data)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(data_paths)} dataset(s)")
    for p in data_paths:
        print(f"  - {Path(p).name}")

    # Get ablation configs
    configs = get_configs(args.type)
    print(f"\nAblation type: {args.type}")
    print(f"Configs to run: {len(configs)}")
    for c in configs:
        print(f"  - {c.name}")

    # Training config
    training_config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        nepochs=args.epochs,
        seed=args.seed,
        mode=args.mode,
    )

    print(f"\nTraining config:")
    print(f"  lr: {training_config.lr}")
    print(f"  batch_size: {training_config.batch_size}")
    print(f"  epochs: {training_config.nepochs}")
    print(f"  mode: {training_config.mode}")

    # Run experiments
    print(f"\nTotal experiments: {len(configs) * len(data_paths)}")
    print(f"Output directory: {args.output}")
    print("\nStarting ablation experiments...\n")

    results = run_ablation_suite(
        configs=configs,
        data_paths=data_paths,
        output_dir=args.output,
        training_config=training_config,
        verbose=not args.quiet,
    )

    # Print summary
    print_results_summary(results)

    # Save final results
    results_path = Path(args.output) / "ablation_results.json"
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
