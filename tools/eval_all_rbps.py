#!/usr/bin/env python
"""Comprehensive saliency evaluation for all available RBP models.

This script runs saliency sanity checks on all trained PrismNet models with:
- Automatic discovery of models and datasets
- Checkpointing to resume interrupted runs
- Progress tracking and logging
- Resource monitoring
- Comprehensive summary generation
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def find_all_models(base_dir: Path) -> List[str]:
    """Find all trained RBP models.

    Args:
        base_dir: Base PrismNet directory

    Returns:
        List of protein names
    """
    model_dir = base_dir / "exp/train_all/out/models"
    models = []

    for model_file in model_dir.glob("*_PrismNet_pu_best.pth"):
        protein = model_file.stem.replace("_PrismNet_pu_best", "")
        models.append(protein)

    return sorted(models)


def check_dataset_exists(protein: str, base_dir: Path) -> bool:
    """Check if dataset exists for protein.

    Args:
        protein: Protein name
        base_dir: Base PrismNet directory

    Returns:
        True if dataset exists
    """
    dataset_path = base_dir / f"data/clip_data/{protein}.h5"
    return dataset_path.exists()


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load evaluation checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint data with sets for completed/failed/skipped
    """
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Checkpoint file is corrupted ({e}). Starting fresh.")
            return {"completed": set(), "failed": set(), "skipped": set()}
        # Convert lists back to sets
        return {
            "completed": set(data.get("completed", [])),
            "failed": set(data.get("failed", [])),
            "skipped": set(data.get("skipped", [])),
        }
    return {"completed": set(), "failed": set(), "skipped": set()}


def save_checkpoint(checkpoint_path: Path, data: Dict):
    """Save evaluation checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        data: Checkpoint data with sets for completed/failed/skipped
    """
    # Convert sets to lists for JSON serialization
    json_data = {
        "completed": sorted(data["completed"]),
        "failed": sorted(data["failed"]),
        "skipped": sorted(data["skipped"]),
    }
    with open(checkpoint_path, "w") as f:
        json.dump(json_data, f, indent=2)


def check_gpu_memory() -> Tuple[float, float]:
    """Check GPU memory usage.

    Returns:
        Tuple of (used_gb, total_gb)
    """
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return used, total
    return 0.0, 0.0


def run_evaluation(
    protein: str,
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    n_samples: int = 100,
    n_random: int = 5,
    timeout: int = 600,
) -> Dict:
    """Run saliency evaluation for a single protein.

    Args:
        protein: Protein name
        model_path: Path to model checkpoint
        dataset_path: Path to dataset
        output_dir: Output directory
        n_samples: Number of samples to test
        n_random: Number of random initializations
        timeout: Timeout in seconds

    Returns:
        Result dictionary
    """
    protein_output = output_dir / protein
    protein_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "python",
        "tools/eval_saliency.py",
        "--model-path", str(model_path),
        "--dataset", str(dataset_path),
        "--n-samples", str(n_samples),
        "--n-random", str(n_random),
        "--test-type", "both",
        "--output-dir", str(protein_output),
    ]

    start_time = time.time()

    # Log file for this protein's output
    log_file = protein_output / "evaluation.log"

    try:
        # Run with output redirected to log file instead of buffering in memory
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                stdout=f,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                timeout=timeout,
            )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Load results
            full_results_path = protein_output / "full_randomization_results.json"
            if full_results_path.exists():
                with open(full_results_path) as f:
                    full_results = json.load(f)

                return {
                    "status": "success",
                    "protein": protein,
                    "elapsed_time": elapsed,
                    "trained_vs_random_ssim": full_results["trained_vs_random"]["ssim_mean"]["mean"],
                    "random_vs_random_ssim": full_results["random_vs_random"]["ssim_mean"]["mean"],
                    "trained_vs_random_spearman": full_results["trained_vs_random"]["spearman_mean"]["mean"],
                    "random_vs_random_spearman": full_results["random_vs_random"]["spearman_mean"]["mean"],
                }
            else:
                return {
                    "status": "success_no_results",
                    "protein": protein,
                    "elapsed_time": elapsed,
                }
        else:
            # Read last 500 chars of log for error message
            error_msg = "Unknown error"
            if log_file.exists():
                with open(log_file) as f:
                    content = f.read()
                    error_msg = content[-500:] if content else "Empty log"

            return {
                "status": "failed",
                "protein": protein,
                "elapsed_time": elapsed,
                "error": error_msg,
                "log_file": str(log_file),
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "protein": protein,
            "elapsed_time": timeout,
        }
    except (FileNotFoundError, PermissionError) as e:
        return {
            "status": "io_error",
            "protein": protein,
            "error": f"{type(e).__name__}: {str(e)}",
        }
    except json.JSONDecodeError as e:
        return {
            "status": "json_error",
            "protein": protein,
            "error": f"Failed to parse results JSON: {str(e)}",
        }
    except KeyError as e:
        return {
            "status": "result_format_error",
            "protein": protein,
            "error": f"Unexpected result format, missing key: {str(e)}",
        }
    except Exception as e:
        # Catch-all for truly unexpected errors
        return {
            "status": "unexpected_error",
            "protein": protein,
            "error": f"{type(e).__name__}: {str(e)}",
        }


def main():
    # Use environment variables with sensible defaults
    base_dir = Path(os.getenv("PRISMNET_DIR", Path.cwd()))
    output_dir = base_dir / "evaluation" / "saliency" / "comprehensive_evaluation"
    checkpoint_path = output_dir / "checkpoint.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all models
    print("Discovering available models...")
    all_proteins = find_all_models(base_dir)
    print(f"Found {len(all_proteins)} trained models")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    completed = checkpoint["completed"]
    failed = checkpoint["failed"]
    skipped = checkpoint["skipped"]

    # Filter proteins to evaluate
    to_evaluate = [p for p in all_proteins if p not in completed and p not in skipped]

    if not to_evaluate:
        print("All proteins already evaluated!")
        return

    print("\nEvaluation Status:")
    print(f"  Completed: {len(completed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Remaining: {len(to_evaluate)}")
    print(f"\nStarting evaluation of {len(to_evaluate)} proteins...")

    # Evaluation parameters
    n_samples = 100
    n_random = 5
    timeout = 600  # 10 minutes per protein

    results = []
    start_time = time.time()

    for i, protein in enumerate(to_evaluate, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(to_evaluate)}] Evaluating: {protein}")
        print(f"{'='*70}")

        # Check dataset
        if not check_dataset_exists(protein, base_dir):
            print(f"⊘ Skipping {protein}: Dataset not found")
            skipped.add(protein)
            save_checkpoint(checkpoint_path, checkpoint)
            continue

        # Check GPU memory
        used_gb, total_gb = check_gpu_memory()
        print(f"GPU Memory: {used_gb:.1f}GB / {total_gb:.1f}GB")

        # Run evaluation
        model_path = base_dir / f"exp/train_all/out/models/{protein}_PrismNet_pu_best.pth"
        dataset_path = base_dir / f"data/clip_data/{protein}.h5"

        result = run_evaluation(
            protein,
            model_path,
            dataset_path,
            output_dir,
            n_samples=n_samples,
            n_random=n_random,
            timeout=timeout,
        )

        results.append(result)

        # Update checkpoint
        if result["status"] == "success":
            print(f"✓ Success: {protein} ({result['elapsed_time']:.1f}s)")
            completed.add(protein)
            failed.discard(protein)
        elif result["status"] == "success_no_results":
            print(f"⚠ Success but no results: {protein}")
            completed.add(protein)
            failed.discard(protein)
        else:
            print(f"✗ {result['status'].title()}: {protein}")
            failed.add(protein)

        save_checkpoint(checkpoint_path, checkpoint)

        # Progress update
        elapsed_total = time.time() - start_time
        avg_time = elapsed_total / i
        remaining = len(to_evaluate) - i
        eta = avg_time * remaining

        print(f"\nProgress: {i}/{len(to_evaluate)} ({i/len(to_evaluate)*100:.1f}%)")
        print(f"Elapsed: {elapsed_total/3600:.1f}h | ETA: {eta/3600:.1f}h")
        print(f"Success: {len(completed)} | Failed: {len(failed)} | Skipped: {len(skipped)}")

    # Final summary
    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("COMPREHENSIVE EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total proteins: {len(all_proteins)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skipped)}")

    # Save final results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "total_proteins": len(all_proteins),
            "completed": len(completed),
            "failed": len(failed),
            "skipped": len(skipped),
            "total_time_hours": total_time / 3600,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
