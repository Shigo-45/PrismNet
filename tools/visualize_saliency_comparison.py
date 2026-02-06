#!/usr/bin/env python3
"""
Visualize saliency maps and HARs side-by-side for plain CNN vs PrismNet.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import sys

def load_saliency_data(filepath, n_samples=10):
    """Load saliency data and parse saliency matrices."""
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                idx = int(parts[0])
                prob = float(parts[1])
                sal_str = parts[2]
                # Parse saliency matrix: "val,val,val,..." format
                # Each line has shape (101, 5) flattened
                sal_values = [float(v) for v in sal_str.split(',') if v.strip()]
                # Pad or trim to expected size
                expected_size = 101 * 5
                if len(sal_values) > expected_size:
                    sal_values = sal_values[:expected_size]
                elif len(sal_values) < expected_size:
                    sal_values.extend([0.0] * (expected_size - len(sal_values)))
                sal_matrix = np.array(sal_values).reshape(101, 5)
                data.append({
                    'idx': idx,
                    'prob': prob,
                    'saliency': sal_matrix
                })
    return data

def load_har_data(filepath, n_samples=10):
    """Load HAR positions."""
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                idx = int(parts[0])
                prob = float(parts[1])
                start = int(parts[2])
                end = int(parts[3])
                data.append({
                    'idx': idx,
                    'prob': prob,
                    'start': start,
                    'end': end
                })
    return data

def plot_comparison(plain_sal, prism_sal, plain_har, prism_har,
                   sample_idx, output_file):
    """Create side-by-side comparison plot for a single sequence."""

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Get data for this sample
    p_sal = plain_sal[sample_idx]['saliency']
    pr_sal = prism_sal[sample_idx]['saliency']
    p_har = plain_har[sample_idx]
    pr_har = prism_har[sample_idx]

    seq_idx = plain_sal[sample_idx]['idx']

    # Title
    fig.suptitle(f'Saliency Map Comparison - Sequence {seq_idx}\n' +
                 f'Plain CNN prob={p_har["prob"]:.3f}, PrismNet prob={pr_har["prob"]:.3f}',
                 fontsize=14, fontweight='bold')

    # === Row 1: Full saliency heatmaps ===
    # Plain CNN
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(p_sal.T, aspect='auto', cmap='YlOrRd',
                     interpolation='nearest')
    ax1.set_title(f'Plain CNN Saliency Map\nHAR: [{p_har["start"]}-{p_har["end"]}]',
                  fontsize=11)
    ax1.set_ylabel('Feature Channel', fontsize=10)
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels(['A', 'C', 'G', 'U', 'icSHAPE'])
    ax1.axvline(p_har['start'], color='blue', linewidth=2, alpha=0.7)
    ax1.axvline(p_har['end'], color='blue', linewidth=2, alpha=0.7)
    ax1.axvspan(p_har['start'], p_har['end'], alpha=0.2, color='blue')
    plt.colorbar(im1, ax=ax1, label='Saliency Score')

    # PrismNet
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pr_sal.T, aspect='auto', cmap='YlOrRd',
                     interpolation='nearest')
    ax2.set_title(f'PrismNet Saliency Map\nHAR: [{pr_har["start"]}-{pr_har["end"]}]',
                  fontsize=11)
    ax2.set_ylabel('Feature Channel', fontsize=10)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['A', 'C', 'G', 'U', 'icSHAPE'])
    ax2.axvline(pr_har['start'], color='red', linewidth=2, alpha=0.7)
    ax2.axvline(pr_har['end'], color='red', linewidth=2, alpha=0.7)
    ax2.axvspan(pr_har['start'], pr_har['end'], alpha=0.2, color='red')
    plt.colorbar(im2, ax=ax2, label='Saliency Score')

    # === Row 2: Sum across channels (total attention per position) ===
    p_sum = p_sal.sum(axis=1)
    pr_sum = pr_sal.sum(axis=1)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(p_sum, color='blue', linewidth=2, label='Plain CNN')
    ax3.axvspan(p_har['start'], p_har['end'], alpha=0.2, color='blue',
                label=f'HAR [{p_har["start"]}-{p_har["end"]}]')
    ax3.set_title('Total Attention per Position', fontsize=11)
    ax3.set_xlabel('Position (nt)', fontsize=10)
    ax3.set_ylabel('Total Saliency', fontsize=10)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(pr_sum, color='red', linewidth=2, label='PrismNet')
    ax4.axvspan(pr_har['start'], pr_har['end'], alpha=0.2, color='red',
                label=f'HAR [{pr_har["start"]}-{pr_har["end"]}]')
    ax4.set_title('Total Attention per Position', fontsize=11)
    ax4.set_xlabel('Position (nt)', fontsize=10)
    ax4.set_ylabel('Total Saliency', fontsize=10)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # === Row 3: Overlay comparison ===
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(p_sum, color='blue', linewidth=2, label='Plain CNN', alpha=0.7)
    ax5.plot(pr_sum, color='red', linewidth=2, label='PrismNet', alpha=0.7)
    ax5.axvspan(p_har['start'], p_har['end'], alpha=0.2, color='blue')
    ax5.axvspan(pr_har['start'], pr_har['end'], alpha=0.2, color='red')
    ax5.set_title('Overlay: Attention Pattern Comparison', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Position (nt)', fontsize=10)
    ax5.set_ylabel('Total Saliency', fontsize=10)
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Add HAR overlap region if any
    har_overlap_start = max(p_har['start'], pr_har['start'])
    har_overlap_end = min(p_har['end'], pr_har['end'])
    if har_overlap_end > har_overlap_start:
        ax5.axvspan(har_overlap_start, har_overlap_end, alpha=0.3,
                   color='purple', label=f'HAR Overlap [{har_overlap_start}-{har_overlap_end}]')
        ax5.legend(loc='upper right', fontsize=10)

    # === Row 4: Per-channel comparison for icSHAPE ===
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(p_sal[:, 4], color='blue', linewidth=2, label='Plain CNN')
    ax6.axvspan(p_har['start'], p_har['end'], alpha=0.2, color='blue')
    ax6.set_title('icSHAPE Feature Saliency', fontsize=11)
    ax6.set_xlabel('Position (nt)', fontsize=10)
    ax6.set_ylabel('Saliency', fontsize=10)
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(pr_sal[:, 4], color='red', linewidth=2, label='PrismNet')
    ax7.axvspan(pr_har['start'], pr_har['end'], alpha=0.2, color='red')
    ax7.set_title('icSHAPE Feature Saliency', fontsize=11)
    ax7.set_xlabel('Position (nt)', fontsize=10)
    ax7.set_ylabel('Saliency', fontsize=10)
    ax7.legend(loc='upper right', fontsize=9)
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def main():
    # File paths
    plain_sal_file = 'exp/plain_cnn_2d1d/out/saliency/SND1_K562_PlainCNN2D1D_pu.sal'
    prism_sal_file = '/home/shigo-45/projects/PrismNet/exp/train_all/out/saliency/SND1_K562_PrismNet_pu.sal'
    plain_har_file = 'exp/plain_cnn_2d1d/out/har/SND1_K562_PlainCNN2D1D_pu.har'
    prism_har_file = '/home/shigo-45/projects/PrismNet/exp/train_all/out/har/SND1_K562_PrismNet_pu.har'

    output_dir = 'exp/plain_cnn_2d1d/visualizations'

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load data (first 100 sequences to choose from)
    print("Loading saliency data...")
    plain_sal = load_saliency_data(plain_sal_file, n_samples=100)
    prism_sal = load_saliency_data(prism_sal_file, n_samples=100)

    print("Loading HAR data...")
    plain_har = load_har_data(plain_har_file, n_samples=100)
    prism_har = load_har_data(prism_har_file, n_samples=100)

    print(f"Loaded {len(plain_sal)} sequences")

    # Calculate HAR overlap for selection
    overlaps = []
    for i in range(len(plain_har)):
        p_start, p_end = plain_har[i]['start'], plain_har[i]['end']
        pr_start, pr_end = prism_har[i]['start'], prism_har[i]['end']
        overlap = max(0, min(p_end, pr_end) - max(p_start, pr_start))
        distance = abs((p_start + p_end)/2 - (pr_start + pr_end)/2)
        overlaps.append((i, overlap, distance))

    # Select representative examples
    # 1. High overlap (models agree)
    high_overlap = sorted(overlaps, key=lambda x: x[1], reverse=True)[:5]

    # 2. Low overlap but close (nearby attention)
    close_but_different = sorted([x for x in overlaps if x[1] < 5 and x[2] < 20],
                                  key=lambda x: x[2])[:5]

    # 3. Far apart (completely different)
    far_apart = sorted([x for x in overlaps if x[1] == 0],
                       key=lambda x: x[2], reverse=True)[:5]

    print("\nGenerating visualizations...")

    # Generate plots
    print("\n=== High Overlap Examples (Models Agree) ===")
    for rank, (idx, overlap, dist) in enumerate(high_overlap):
        output_file = f'{output_dir}/high_overlap_{rank+1}_seq{plain_har[idx]["idx"]}.png'
        print(f"Sample {idx}: overlap={overlap}nt, distance={dist:.1f}nt")
        plot_comparison(plain_sal, prism_sal, plain_har, prism_har, idx, output_file)

    print("\n=== Close But Different Examples ===")
    for rank, (idx, overlap, dist) in enumerate(close_but_different):
        output_file = f'{output_dir}/close_different_{rank+1}_seq{plain_har[idx]["idx"]}.png'
        print(f"Sample {idx}: overlap={overlap}nt, distance={dist:.1f}nt")
        plot_comparison(plain_sal, prism_sal, plain_har, prism_har, idx, output_file)

    print("\n=== Far Apart Examples (Models Disagree) ===")
    for rank, (idx, overlap, dist) in enumerate(far_apart):
        output_file = f'{output_dir}/far_apart_{rank+1}_seq{plain_har[idx]["idx"]}.png'
        print(f"Sample {idx}: overlap={overlap}nt, distance={dist:.1f}nt")
        plot_comparison(plain_sal, prism_sal, plain_har, prism_har, idx, output_file)

    print(f"\n✓ Generated 15 comparison plots in {output_dir}/")
    print(f"\nVisualization categories:")
    print(f"  - high_overlap_*.png: Models agree on attention location")
    print(f"  - close_different_*.png: Similar but distinct attention patterns")
    print(f"  - far_apart_*.png: Completely different attention locations")

if __name__ == '__main__':
    main()
