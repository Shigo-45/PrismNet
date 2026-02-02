#!/usr/bin/env python
"""Batch runner for saliency evaluations across multiple protein models.

This script runs comprehensive saliency sanity checks on multiple trained
PrismNet models and generates comparative analysis.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List

# Representative subset of proteins from different families and cell lines
PROTEINS_TO_EVALUATE = [
    # RNA binding proteins - different families
    "TIA1_Hela",
    "ELAVL1_Hela",
    "HNRNPC_Hela",
    "PTBP1_Hela",

    # IGF2BP family
    "IGF2BP1_K562",
    "IGF2BP2_HEK293",
    "IGF2BP3_HEK293",

    # Splicing factors
    "SRSF1_HepG2",
    "U2AF2_Hela",
    "SF3B1_K562",

    # m6A writers/readers
    "METTL3_Hela",
    "METTL14_Hela",
    "YTHDF2_Hela",

    # Translation factors
    "EIF4A3_Hela",
    "EIF3G_K562",

    # Other interesting proteins
    "FUS_HEK293",
    "TARDBP_K562",
    "DDX6_K562",
    "PUM1_K562",
    "AGO_HEK293",
]


def find_model_and_dataset(protein_name: str, base_dir: Path) -> tuple:
    """Find model checkpoint and corresponding dataset.

    Args:
        protein_name: Protein identifier (e.g., TIA1_Hela)
        base_dir: Base PrismNet directory

    Returns:
        Tuple of (model_path, dataset_path) or (None, None) if not found
    """
    # Look for model in train_all first, then train_one
    model_path = base_dir / f"exp/train_all/out/models/{protein_name}_PrismNet_pu_best.pth"
    if not model_path.exists():
        model_path = base_dir / f"exp/train_one/out/models/{protein_name}_PrismNet_pu_best.pth"

    if not model_path.exists():
        return None, None

    # Look for dataset
    dataset_path = base_dir / f"data/clip_data/{protein_name}.h5"
    if not dataset_path.exists():
        return model_path, None

    return model_path, dataset_path


def run_evaluation(
    protein_name: str,
    model_path: Path,
    dataset_path: Path,
    output_base: Path,
    n_samples: int = 100,
    n_random: int = 5,
) -> Dict:
    """Run saliency evaluation for a single protein.

    Args:
        protein_name: Protein identifier
        model_path: Path to model checkpoint
        dataset_path: Path to dataset
        output_base: Base output directory
        n_samples: Number of samples to test
        n_random: Number of random initializations

    Returns:
        Dictionary with evaluation results and status
    """
    output_dir = output_base / protein_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "tools/eval_saliency.py",
        "--model-path", str(model_path),
        "--dataset", str(dataset_path),
        "--n-samples", str(n_samples),
        "--n-random", str(n_random),
        "--test-type", "both",
        "--output-dir", str(output_dir),
    ]

    print(f"\n{'='*70}")
    print(f"Evaluating: {protein_name}")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per protein
        )

        if result.returncode == 0:
            print(f"✓ Success: {protein_name}")

            # Load results
            full_results_path = output_dir / "full_randomization_results.json"
            cascading_results_path = output_dir / "cascading_randomization_results.json"

            results = {"status": "success", "protein": protein_name}

            if full_results_path.exists():
                with open(full_results_path) as f:
                    results["full_test"] = json.load(f)

            if cascading_results_path.exists():
                with open(cascading_results_path) as f:
                    results["cascading_test"] = json.load(f)

            return results
        else:
            print(f"✗ Failed: {protein_name}")
            print(f"Error: {result.stderr}")
            return {
                "status": "failed",
                "protein": protein_name,
                "error": result.stderr,
            }

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {protein_name}")
        return {"status": "timeout", "protein": protein_name}

    except Exception as e:
        print(f"✗ Exception: {protein_name} - {e}")
        return {"status": "error", "protein": protein_name, "error": str(e)}


def generate_summary_report(results: List[Dict], output_path: Path):
    """Generate a summary report across all evaluated proteins.

    Args:
        results: List of evaluation results
        output_path: Where to save the summary
    """
    summary = {
        "total_proteins": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "proteins": {},
    }

    for result in results:
        if result["status"] == "success":
            protein = result["protein"]

            # Extract key metrics
            full_test = result.get("full_test", {})
            tvr = full_test.get("trained_vs_random", {})
            rvr = full_test.get("random_vs_random", {})

            summary["proteins"][protein] = {
                "trained_vs_random_ssim": tvr.get("ssim_mean", {}).get("mean"),
                "random_vs_random_ssim": rvr.get("ssim_mean", {}).get("mean"),
                "trained_vs_random_spearman": tvr.get("spearman_mean", {}).get("mean"),
                "random_vs_random_spearman": rvr.get("spearman_mean", {}).get("mean"),
            }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")
    print(f"Total proteins evaluated: {summary['total_proteins']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"\nSummary saved to: {output_path}")


def main():
    base_dir = Path("/home/shigo-45/projects/PrismNet")
    eval_dir = Path("/home/shigo-45/projects/PrismNet-eval-saliency")
    output_base = eval_dir / "evaluation" / "saliency" / "batch_evaluation"

    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Starting batch evaluation of {len(PROTEINS_TO_EVALUATE)} proteins...")
    print(f"Output directory: {output_base}")

    results = []
    skipped = []

    for protein in PROTEINS_TO_EVALUATE:
        model_path, dataset_path = find_model_and_dataset(protein, base_dir)

        if model_path is None:
            print(f"⊘ Skipping {protein}: Model not found")
            skipped.append(protein)
            continue

        if dataset_path is None:
            print(f"⊘ Skipping {protein}: Dataset not found")
            skipped.append(protein)
            continue

        result = run_evaluation(
            protein,
            model_path,
            dataset_path,
            output_base,
            n_samples=100,
            n_random=5,
        )
        results.append(result)

    # Generate summary
    summary_path = output_base / "evaluation_summary.json"
    generate_summary_report(results, summary_path)

    if skipped:
        print(f"\nSkipped proteins ({len(skipped)}): {', '.join(skipped)}")

    print(f"\n{'='*70}")
    print("BATCH EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
