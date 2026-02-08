#!/usr/bin/env python
"""Compute saliency maps and high attention regions (HAR) for ablation models.

This script generates .sal and .har files for all ablation configurations
across specified datasets. It reuses existing infrastructure from train_loop.py
and supports incremental processing with checkpointing.

Usage:
    python tools/compute_ablation_saliency.py \
        --datasets TIA1_Hela,YTHDF2_Hela \
        --configs full_model,no_se,minimal \
        --models-dir evaluation/ablation/models \
        --output evaluation/ablation_saliency

Output files:
    - evaluation/ablation_saliency/saliency/{dataset}_{config}.sal
    - evaluation/ablation_saliency/har/{dataset}_{config}.har
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

# Add parent directory to path to import prismnet modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from prismnet.loader import SeqicSHAPE
from prismnet.utils import datautils
from prismnet.model import GuidedBackpropSmoothGrad
from prismnet_eval.ablation.config import (
    AblationConfig,
    LEAVE_ONE_OUT_CONFIGS,
    CUMULATIVE_CONFIGS,
)
from prismnet_eval.ablation.variants import create_ablated_model
from prismnet_eval.ablation.baselines import PlainCNN2D1D


# Config mapping for easy lookup - combine both config sets
CONFIG_MAP = {}
for config in LEAVE_ONE_OUT_CONFIGS + CUMULATIVE_CONFIGS:
    CONFIG_MAP[config.name] = config
# Add baseline config
CONFIG_MAP['plain_cnn_2d1d'] = 'baseline'


class SaliencyComputer:
    """Computes saliency maps and HARs for ablation models."""

    def __init__(
        self,
        models_dir: str,
        output_dir: str,
        data_dir: str = "data",
        batch_size: Optional[int] = None,
        num_workers: int = 4,
        use_amp: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize SaliencyComputer.

        Args:
            models_dir: Directory containing trained model checkpoints
            output_dir: Output directory for saliency files
            data_dir: Directory containing HDF5 data files
            batch_size: Batch size for inference (auto-detected if None)
            num_workers: Number of DataLoader workers for parallel loading
            use_amp: Use automatic mixed precision for faster computation
            device: Device to use (cuda/cpu), auto-detect if None
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.use_amp = use_amp

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Auto-detect optimal batch size if not specified
        self.batch_size = batch_size or self._get_optimal_batch_size()

        # Create output directories
        self.saliency_dir = self.output_dir / "saliency"
        self.har_dir = self.output_dir / "har"
        self.saliency_dir.mkdir(parents=True, exist_ok=True)
        self.har_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"DataLoader workers: {self.num_workers}")
        print(f"Mixed precision (AMP): {self.use_amp}")
        print(f"Models directory: {self.models_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Data directory: {self.data_dir}")

    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on available GPU memory.

        Returns:
            Optimal batch size for current hardware
        """
        if not torch.cuda.is_available():
            return 32  # Conservative for CPU

        try:
            gpu_mem = torch.cuda.get_device_properties(self.device).total_memory

            if gpu_mem > 20e9:  # >20GB (A100, V100)
                return 256
            elif gpu_mem > 10e9:  # >10GB (RTX 3080, etc)
                return 128
            else:  # 8GB or less
                return 64
        except Exception:
            return 64  # Safe default

    def _load_model(self, dataset: str, config_name: str) -> torch.nn.Module:
        """Load trained model for given dataset and configuration.

        Args:
            dataset: Dataset name (e.g., "TIA1_Hela")
            config_name: Configuration name (e.g., "full_model", "no_se")

        Returns:
            Loaded PyTorch model

        Raises:
            FileNotFoundError: If model checkpoint doesn't exist
        """
        # Determine model path based on config type
        if config_name == 'plain_cnn_2d1d':
            # Baseline model from baselines_full directory
            model_path = Path("evaluation/baselines_full/models") / f"{dataset}_{config_name}.pth"
        else:
            # Ablation model from ablation directory
            model_path = self.models_dir / f"{dataset}_{config_name}.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Create model architecture
        if config_name == 'plain_cnn_2d1d':
            model = PlainCNN2D1D(mode='pu')
        else:
            config = CONFIG_MAP[config_name]
            model = create_ablated_model(config, mode='pu', base_channel=8)

        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        return model

    def _create_test_loader(self, dataset: str) -> torch.utils.data.DataLoader:
        """Create test data loader for given dataset.

        Args:
            dataset: Dataset name (e.g., "TIA1_Hela")

        Returns:
            PyTorch DataLoader for test set
        """
        data_path = self.data_dir / f"{dataset}.h5"

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        test_dataset = SeqicSHAPE(data_path, is_test=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return test_loader

    def compute_saliency(
        self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader
    ) -> List[Dict]:
        """Compute saliency maps for all test samples.

        This function mirrors compute_saliency() from train_loop.py but returns
        structured data instead of writing to file directly. Uses mixed precision
        (AMP) for faster computation while preserving FP32 precision in outputs.

        Args:
            model: Trained PyTorch model
            test_loader: DataLoader for test set

        Returns:
            List of dicts with keys: idx, prob, saliency (shape: 101x5)
        """
        model.eval()
        sgrad = GuidedBackpropSmoothGrad(model, device=self.device)

        results = []

        for batch_idx, (x0, y0) in enumerate(test_loader):
            X = x0.float().to(self.device)
            Y = y0.to(self.device).float()

            # Forward pass with optional AMP (faster)
            with torch.no_grad():
                if self.use_amp and torch.cuda.is_available():
                    with autocast():
                        output = model(X)
                        prob = torch.sigmoid(output)
                else:
                    output = model(X)
                    prob = torch.sigmoid(output)

                # Convert to FP32 for storage (no precision loss)
                p_np = prob.cpu().float().numpy().squeeze(-1)

            # Gradient computation with optional AMP
            if self.use_amp and torch.cuda.is_available():
                with autocast():
                    guided_saliency = sgrad.get_batch_gradients(X, Y)
                # Convert to FP32 for storage
                guided_saliency = guided_saliency.float()
            else:
                guided_saliency = sgrad.get_batch_gradients(X, Y)

            N, NS, _, _ = guided_saliency.shape  # (N, 101, 1, 5)

            # Store results
            for i in range(N):
                inr = batch_idx * self.batch_size + i
                sal = np.squeeze(guided_saliency[i].cpu().numpy())  # Shape: (101, 5)
                results.append({
                    'idx': inr,
                    'prob': float(p_np[i]),
                    'saliency': sal,
                })

        return results

    def compute_har_from_saliency(self, saliency_results: List[Dict]) -> List[Dict]:
        """Compute HAR from already-computed saliency values.

        This ensures consistency - HAR is derived from the exact saliency values
        saved to .sal files. No separate gradient computation with different random
        noise, which was causing mismatches between visualizations and HAR overlays.

        Args:
            saliency_results: List of saliency results from compute_saliency()

        Returns:
            List of dicts with keys: idx, prob, start, end
        """
        L = 20  # HAR window length
        har_results = []

        for result in saliency_results:
            sal = result['saliency']  # Shape: (101, 5)
            # Sum across all channels to get total attention per position
            attention = sal.sum(axis=1)  # Shape: (101,)
            highest_ind = int(np.argmax(attention))

            har_results.append({
                'idx': result['idx'],
                'prob': result['prob'],
                'start': highest_ind,
                'end': highest_ind + L,
            })

        return har_results

    def save_saliency(self, results: List[Dict], output_path: Path):
        """Save saliency results to .sal file.

        Format: idx\tprob\tsal1,sal2,...,sal505
        Where sal values are flattened 101x5 matrix (comma-separated)

        Args:
            results: List of saliency results from compute_saliency()
            output_path: Path to output .sal file
        """
        with open(output_path, 'w') as f:
            for r in results:
                # Flatten saliency matrix and format as comma-separated string
                sal_flat = r['saliency'].flatten()
                str_sal = ','.join([f"{x:.6f}" for x in sal_flat])
                f.write(f"{r['idx']}\t{r['prob']:.6f}\t{str_sal}\n")

    def save_har(self, results: List[Dict], output_path: Path):
        """Save HAR results to .har file.

        Format: idx\tprob\tstart\tend

        Args:
            results: List of HAR results from compute_har()
            output_path: Path to output .har file
        """
        with open(output_path, 'w') as f:
            for r in results:
                f.write(f"{r['idx']}\t{r['prob']:.6f}\t{r['start']}\t{r['end']}\n")

    def process_dataset_config(
        self, dataset: str, config_name: str, skip_existing: bool = True
    ) -> bool:
        """Process single dataset-config combination.

        Args:
            dataset: Dataset name
            config_name: Configuration name
            skip_existing: Skip if output files already exist

        Returns:
            True if processing completed, False if skipped
        """
        sal_path = self.saliency_dir / f"{dataset}_{config_name}.sal"
        har_path = self.har_dir / f"{dataset}_{config_name}.har"

        # Check if outputs already exist
        if skip_existing and sal_path.exists() and har_path.exists():
            return False

        try:
            # Load model
            model = self._load_model(dataset, config_name)

            # Create test loader
            test_loader = self._create_test_loader(dataset)

            # Compute saliency
            saliency_results = self.compute_saliency(model, test_loader)
            self.save_saliency(saliency_results, sal_path)

            # Compute HAR from saliency (no separate gradient computation)
            har_results = self.compute_har_from_saliency(saliency_results)
            self.save_har(har_results, har_path)

            return True

        except Exception as e:
            print(f"\nError processing {dataset} with {config_name}: {e}")
            # Clean up partial files
            if sal_path.exists():
                sal_path.unlink()
            if har_path.exists():
                har_path.unlink()
            raise

    def batch_process(
        self,
        datasets: List[str],
        configs: List[str],
        checkpoint_path: Optional[Path] = None,
        skip_existing: bool = True,
    ) -> Dict:
        """Process multiple datasets and configurations with progress tracking.

        Args:
            datasets: List of dataset names
            configs: List of configuration names
            checkpoint_path: Path to checkpoint file for resuming
            skip_existing: Skip already processed combinations

        Returns:
            Dictionary with processing statistics
        """
        # Load checkpoint if exists
        completed = set()
        if checkpoint_path and checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                completed = set(tuple(x) for x in checkpoint.get('completed', []))
            print(f"Loaded checkpoint: {len(completed)} already completed")

        # Generate all combinations
        total = len(datasets) * len(configs)
        processed = 0
        skipped = 0
        failed = []

        # Progress bar
        with tqdm(total=total, desc="Processing") as pbar:
            for dataset in datasets:
                for config_name in configs:
                    pbar.set_description(f"{dataset}/{config_name}")

                    # Check checkpoint
                    if (dataset, config_name) in completed:
                        skipped += 1
                        pbar.update(1)
                        continue

                    try:
                        success = self.process_dataset_config(
                            dataset, config_name, skip_existing
                        )
                        if success:
                            processed += 1
                            completed.add((dataset, config_name))
                        else:
                            skipped += 1

                    except Exception as e:
                        print(f"\nFailed: {dataset}/{config_name}: {e}")
                        failed.append((dataset, config_name, str(e)))

                    # Update checkpoint
                    if checkpoint_path:
                        with open(checkpoint_path, 'w') as f:
                            json.dump({
                                'completed': list(completed),
                                'failed': failed,
                            }, f, indent=2)

                    pbar.update(1)

        # Summary
        stats = {
            'total': total,
            'processed': processed,
            'skipped': skipped,
            'failed': len(failed),
            'failed_items': failed,
        }

        print("\n" + "="*60)
        print("Processing Summary")
        print("="*60)
        print(f"Total combinations: {total}")
        print(f"Newly processed: {processed}")
        print(f"Skipped (existing): {skipped}")
        print(f"Failed: {len(failed)}")
        if failed:
            print("\nFailed items:")
            for dataset, config, error in failed:
                print(f"  - {dataset}/{config}: {error}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute saliency maps and HARs for ablation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--datasets',
        type=str,
        required=True,
        help='Comma-separated dataset names or path to .list file'
    )
    parser.add_argument(
        '--configs',
        type=str,
        default='full_model,no_se,no_res2d_skip,no_res1d_skip,minimal,plain_cnn_2d1d',
        help='Comma-separated config names (default: all 6 configs)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='evaluation/ablation/models',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing HDF5 data files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation/ablation_saliency',
        help='Output directory for saliency files'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for inference (auto-detected if not specified)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader workers for parallel data loading (default: 4)'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable mixed precision (slower but max precision)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint file for resuming'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Recompute even if output files exist'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (auto-detect if not specified)'
    )

    args = parser.parse_args()

    # Parse datasets
    if args.datasets.endswith('.list'):
        with open(args.datasets, 'r') as f:
            datasets = [line.strip() for line in f if line.strip()]
    else:
        datasets = [d.strip() for d in args.datasets.split(',')]

    # Parse configs
    configs = [c.strip() for c in args.configs.split(',')]

    # Validate configs
    valid_configs = set(CONFIG_MAP.keys())
    invalid = [c for c in configs if c not in valid_configs]
    if invalid:
        print(f"Error: Invalid configs: {invalid}")
        print(f"Valid configs: {sorted(valid_configs)}")
        sys.exit(1)

    print("="*60)
    print("Ablation Saliency Computation")
    print("="*60)
    print(f"Datasets: {len(datasets)}")
    print(f"Configs: {len(configs)}")
    print(f"Total combinations: {len(datasets) * len(configs)}")
    print()

    # Create computer and run
    computer = SaliencyComputer(
        models_dir=args.models_dir,
        output_dir=args.output,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        device=args.device,
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    stats = computer.batch_process(
        datasets=datasets,
        configs=configs,
        checkpoint_path=checkpoint_path,
        skip_existing=not args.no_skip_existing,
    )

    # Save final stats
    stats_path = Path(args.output) / 'processing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to: {stats_path}")


if __name__ == '__main__':
    main()
