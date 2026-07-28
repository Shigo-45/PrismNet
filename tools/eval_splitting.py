#!/usr/bin/env python
"""
CLI tool for homology-aware data splitting evaluation.

Analyzes existing splits and creates new splits using CD-HIT and/or DataSAIL
to minimize homology leakage between train and test sets.
"""

import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prismnet_eval.splitting import (
    analyze_existing_split,
    create_fasta_from_h5,
    cluster_sequences,
    split_by_clusters,
    split_with_datasail,
    create_homology_report,
)


def create_split_h5(
    original_h5: Path,
    output_h5: Path,
    train_seq_ids: list,
    test_seq_ids: list
):
    """
    Create new H5 file with specified train/test split.

    Args:
        original_h5: Path to original H5 file
        output_h5: Path for new H5 file
        train_seq_ids: List of sequence IDs for training
        test_seq_ids: List of sequence IDs for testing
    """
    # Load original data
    with h5py.File(original_h5, "r") as f_in:
        # Get all sequences (train + test from original)
        X_train_orig = f_in["X_train"][:]
        Y_train_orig = f_in["Y_train"][:]
        X_test_orig = f_in["X_test"][:]
        Y_test_orig = f_in["Y_test"][:]

        # Combine all data
        X_all = np.concatenate([X_train_orig, X_test_orig], axis=0)
        Y_all = np.concatenate([Y_train_orig, Y_test_orig], axis=0)

        # Create ID to index mapping
        all_ids = [f"X_train_{i}" for i in range(len(X_train_orig))] + \
                  [f"X_test_{i}" for i in range(len(X_test_orig))]

        id_to_idx = {seq_id: idx for idx, seq_id in enumerate(all_ids)}

        # Get indices for new split
        train_indices = [id_to_idx[seq_id] for seq_id in train_seq_ids if seq_id in id_to_idx]
        dropped = len(train_seq_ids) - len(train_indices)
        if dropped > 0:
            print(f"WARNING: {dropped} train sequences had unrecognized IDs and were dropped")

        test_indices = [id_to_idx[seq_id] for seq_id in test_seq_ids if seq_id in id_to_idx]
        dropped = len(test_seq_ids) - len(test_indices)
        if dropped > 0:
            print(f"WARNING: {dropped} test sequences had unrecognized IDs and were dropped")

        # Create new split
        X_train_new = X_all[train_indices]
        Y_train_new = Y_all[train_indices]
        X_test_new = X_all[test_indices]
        Y_test_new = Y_all[test_indices]

        # Save to new H5 file
        output_h5.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_h5, "w") as f_out:
            f_out.create_dataset("X_train", data=X_train_new)
            f_out.create_dataset("Y_train", data=Y_train_new)
            f_out.create_dataset("X_test", data=X_test_new)
            f_out.create_dataset("Y_test", data=Y_test_new)

            # Copy validation set if exists
            if "X_valid" in f_in:
                f_out.create_dataset("X_valid", data=f_in["X_valid"][:])
                f_out.create_dataset("Y_valid", data=f_in["Y_valid"][:])

            # Copy metadata if exists
            if "experiment" in f_in:
                f_out.create_dataset("experiment", data=f_in["experiment"][:])

    print(f"Created new H5 file: {output_h5}")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")


def process_dataset(
    h5_path: Path,
    methods: list,
    output_dir: Path,
    identity: float,
    analyze_only: bool
):
    """
    Process a single dataset: analyze and optionally create new splits.

    Args:
        h5_path: Path to H5 dataset
        methods: List of methods to use ('cdhit', 'datasail')
        output_dir: Output directory for results
        identity: Sequence identity threshold
        analyze_only: If True, only analyze existing split
    """
    print(f"\n{'='*80}")
    print(f"Processing: {h5_path.name}")
    print(f"{'='*80}\n")

    # Step 1: Analyze existing split
    print("Step 1: Analyzing existing split...")
    original_stats = analyze_existing_split(
        h5_path,
        identity_threshold=identity,
        sample_size=10000  # Sample for speed
    )

    print("\nOriginal split analysis:")
    print(f"  Train sequences: {original_stats['n_train']}")
    print(f"  Test sequences: {original_stats['n_test']}")
    print(f"  Max identity: {original_stats['max_identity']:.3f}")
    print(f"  Mean identity: {original_stats['mean_identity']:.3f}")
    print(f"  Pairs above {identity}: {original_stats['n_pairs_above_threshold']} "
          f"({original_stats['leakage_fraction']*100:.2f}%)")

    results = {"original": original_stats}

    if analyze_only:
        print("\nAnalysis-only mode. Skipping split creation.")
        return results

    # Step 2: Create FASTA file
    print("\nStep 2: Creating FASTA file...")
    fasta_path = create_fasta_from_h5(h5_path, output_dir)

    # Step 3: Process each method
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Method: {method.upper()}")
        print(f"{'='*80}\n")

        if method == "cdhit":
            # Run CD-HIT clustering
            print("Running CD-HIT clustering...")
            clusters = cluster_sequences(fasta_path, identity=identity)

            print(f"Found {len(set(clusters.values()))} clusters from {len(clusters)} sequences")

            # Split by clusters
            train_ids, test_ids = split_by_clusters(clusters, test_fraction=0.2)

            # Create new H5 file
            output_h5 = output_dir / f"{h5_path.stem}_cdhit{int(identity*100)}.h5"
            create_split_h5(h5_path, output_h5, train_ids, test_ids)

            # Analyze new split
            print("\nAnalyzing CD-HIT split...")
            cdhit_stats = analyze_existing_split(output_h5, identity_threshold=identity, sample_size=10000)
            results["cdhit"] = cdhit_stats

            print("\nCD-HIT split analysis:")
            print(f"  Max identity: {cdhit_stats['max_identity']:.3f}")
            print(f"  Mean identity: {cdhit_stats['mean_identity']:.3f}")
            print(f"  Pairs above {identity}: {cdhit_stats['n_pairs_above_threshold']} "
                  f"({cdhit_stats['leakage_fraction']*100:.2f}%)")

        elif method == "datasail":
            # Run DataSAIL
            print("Running DataSAIL...")
            try:
                train_ids, test_ids = split_with_datasail(
                    fasta_path,
                    identity=identity,
                    test_fraction=0.2
                )

                # Create new H5 file
                output_h5 = output_dir / f"{h5_path.stem}_datasail{int(identity*100)}.h5"
                create_split_h5(h5_path, output_h5, train_ids, test_ids)

                # Analyze new split
                print("\nAnalyzing DataSAIL split...")
                datasail_stats = analyze_existing_split(output_h5, identity_threshold=identity, sample_size=10000)
                results["datasail"] = datasail_stats

                print("\nDataSAIL split analysis:")
                print(f"  Max identity: {datasail_stats['max_identity']:.3f}")
                print(f"  Mean identity: {datasail_stats['mean_identity']:.3f}")
                print(f"  Pairs above {identity}: {datasail_stats['n_pairs_above_threshold']} "
                      f"({datasail_stats['leakage_fraction']*100:.2f}%)")

            except Exception as e:
                print(f"DataSAIL failed: {e}")
                print("Skipping DataSAIL method.")

    # Step 4: Create comparison report
    print(f"\n{'='*80}")
    print("Comparison Report")
    print(f"{'='*80}\n")

    report = create_homology_report(
        original_stats,
        cdhit=results.get("cdhit"),
        datasail=results.get("datasail")
    )

    # Print summary
    print("Leakage reduction compared to original:")
    for method in ["cdhit", "datasail"]:
        if method in results:
            improvement_key = f"{method}_improvement"
            if improvement_key in report["summary"]:
                improvement = report["summary"][improvement_key] * 100
                print(f"  {method.upper()}: {improvement:.1f}% reduction in leakage")

    # Save report
    report_path = output_dir / f"{h5_path.stem}_homology_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate homology-aware data splitting methods"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Paths to H5 dataset files"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["cdhit", "datasail", "both"],
        default=["both"],
        help="Splitting methods to use"
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.8,
        help="Sequence identity threshold (default: 0.8)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/homology",
        help="Output directory for results"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing splits, don't create new ones"
    )

    args = parser.parse_args()

    # Parse methods
    if "both" in args.methods:
        methods = ["cdhit", "datasail"]
    else:
        methods = args.methods

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    all_results = {}

    for dataset_path in args.datasets:
        h5_path = Path(dataset_path)

        if not h5_path.exists():
            print(f"Warning: {h5_path} not found. Skipping.")
            continue

        try:
            results = process_dataset(
                h5_path,
                methods,
                output_dir,
                args.identity,
                args.analyze_only
            )

            all_results[h5_path.name] = results

        except Exception as e:
            print(f"Error processing {h5_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save combined results
    if all_results:
        summary_path = output_dir / "splitting_evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"All results saved to: {summary_path}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
