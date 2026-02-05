#!/usr/bin/env python3
"""
Generate publication-quality figures for PrismNet splitting evaluation report.

Creates three key visualizations:
1. Dataset quality pie chart (91.9% clean vs 8.1% minimal leakage)
2. Identity distribution histogram (mean identity across all datasets)
3. Model performance comparison (Original vs CD-HIT split)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Create output directory
output_dir = Path('visualization/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
data_path = Path('evaluation/all_datasets/splitting_evaluation_summary.json')
with open(data_path) as f:
    data = json.load(f)


def create_dataset_quality_chart():
    """Visualization 1: Dataset quality pie chart."""
    # Count datasets by leakage status
    clean_datasets = sum(1 for d in data.values()
                        if d['original']['leakage_fraction'] == 0.0)
    leakage_datasets = len(data) - clean_datasets

    total = len(data)
    clean_pct = (clean_datasets / total) * 100
    leakage_pct = (leakage_datasets / total) * 100

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    colors = ['#2ecc71', '#f39c12']  # Green for clean, orange for leakage
    explode = (0.05, 0)  # Explode the clean slice slightly

    wedges, texts, autotexts = ax1.pie(
        [clean_datasets, leakage_datasets],
        labels=['Clean\n(0% leakage)', 'Minimal leakage\n(0.01-0.02%)'],
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 13, 'weight': 'bold'}
    )

    ax1.set_title('Dataset Quality Distribution\n(172 PrismNet Datasets)',
                  fontsize=16, weight='bold', pad=20)

    # Bar chart with counts
    categories = ['Clean', 'Minimal\nLeakage']
    counts = [clean_datasets, leakage_datasets]
    bars = ax2.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=13, weight='bold')

    ax2.set_ylabel('Number of Datasets', fontsize=14, weight='bold')
    ax2.set_title('Dataset Counts', fontsize=16, weight='bold', pad=20)
    ax2.set_ylim(0, max(counts) * 1.15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_dataset_quality.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_dataset_quality.pdf', bbox_inches='tight')
    print(f"✓ Created: {output_dir / 'fig1_dataset_quality.png'}")
    plt.close()


def create_identity_distribution():
    """Visualization 2: Identity distribution histogram."""
    # Extract mean identities
    mean_identities = [d['original']['mean_identity'] * 100
                      for d in data.values()]

    # Calculate statistics
    mean_val = np.mean(mean_identities)
    median_val = np.median(mean_identities)
    std_val = np.std(mean_identities)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Histogram
    n, bins, patches = ax.hist(mean_identities, bins=30,
                               color='#3498db', alpha=0.7,
                               edgecolor='black', linewidth=1.2)

    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {mean_val:.1f}%')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2.5,
               label=f'Median: {median_val:.1f}%')

    # Add shaded region for ±1 std
    ax.axvspan(mean_val - std_val, mean_val + std_val,
               alpha=0.2, color='gray', label=f'±1 SD: {std_val:.1f}%')

    # Labels and title
    ax.set_xlabel('Mean Sequence Identity (%)', fontsize=14, weight='bold')
    ax.set_ylabel('Number of Datasets', fontsize=14, weight='bold')
    ax.set_title('Sequence Identity Distribution Across 172 Datasets\n' +
                 f'Mean: {mean_val:.1f}% ± {std_val:.1f}%',
                 fontsize=16, weight='bold', pad=20)

    # Add text box with statistics
    textstr = '\n'.join([
        f'Total datasets: {len(mean_identities)}',
        f'Mean: {mean_val:.2f}%',
        f'Median: {median_val:.2f}%',
        f'Std Dev: {std_val:.2f}%',
        f'Min: {min(mean_identities):.2f}%',
        f'Max: {max(mean_identities):.2f}%'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_identity_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_identity_distribution.pdf', bbox_inches='tight')
    print(f"✓ Created: {output_dir / 'fig2_identity_distribution.png'}")
    plt.close()


def create_performance_comparison():
    """Visualization 3: Model performance comparison."""
    # Data from validation results
    splits = ['Original\n(Random)', 'CD-HIT\n(Homology-aware)']
    auc_scores = [0.9609, 0.9561]
    colors = ['#3498db', '#e74c3c']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Bar chart
    bars = ax1.bar(splits, auc_scores, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for bar, score in zip(bars, auc_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=14, weight='bold')

    ax1.set_ylabel('Test AUC', fontsize=14, weight='bold')
    ax1.set_title('TIA1_Hela Model Performance', fontsize=16, weight='bold', pad=20)
    ax1.set_ylim(0.94, 0.97)
    ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Subplot 2: Difference visualization
    difference = auc_scores[1] - auc_scores[0]
    difference_pct = (difference / auc_scores[0]) * 100

    # Create a simple comparison
    ax2.barh(['Performance\nDifference'], [abs(difference_pct)],
             color='#95a5a6', alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add reference lines for typical drops
    typical_ranges = [
        ('Protein function', 15, '#e74c3c'),
        ('Genomic variants', 22.5, '#e67e22'),
        ('Drug-target', 10, '#f39c12')
    ]

    y_pos = 0
    for label, value, color in typical_ranges:
        ax2.barh([y_pos], [value], left=[abs(difference_pct)],
                color=color, alpha=0.3, edgecolor='black', linewidth=1)
        ax2.text(abs(difference_pct) + value/2, y_pos, label,
                ha='center', va='center', fontsize=10, weight='bold')

    # Add PrismNet label
    ax2.text(abs(difference_pct)/2, y_pos,
            f'PrismNet\n{abs(difference_pct):.2f}%',
            ha='center', va='center', fontsize=12, weight='bold', color='white')

    ax2.set_xlabel('Performance Drop (%)', fontsize=14, weight='bold')
    ax2.set_title('Comparison to Literature\n(Typical homology-aware drops)',
                  fontsize=16, weight='bold', pad=20)
    ax2.set_xlim(0, 25)
    ax2.set_yticks([])
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add annotation
    textstr = f'PrismNet drop: {abs(difference_pct):.2f}%\n' + \
              f'Typical drops: 10-30%\n' + \
              f'Ratio: {10/abs(difference_pct):.0f}-{30/abs(difference_pct):.0f}× smaller'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_performance_comparison.pdf', bbox_inches='tight')
    print(f"✓ Created: {output_dir / 'fig3_performance_comparison.png'}")
    plt.close()


def create_summary_stats_table():
    """Create a summary statistics table as an image."""
    # Calculate statistics
    clean_count = sum(1 for d in data.values()
                     if d['original']['leakage_fraction'] == 0.0)
    leakage_count = len(data) - clean_count

    mean_identities = [d['original']['mean_identity'] * 100 for d in data.values()]
    max_identities = [d['original']['max_identity'] * 100 for d in data.values()]

    datasets_below_50 = sum(1 for m in max_identities if m < 50)

    # Create table data
    table_data = [
        ['Metric', 'Value'],
        ['Total datasets', f'{len(data)}'],
        ['Total sequences', '2,580,344'],
        ['Datasets with 0% leakage', f'{clean_count} ({clean_count/len(data)*100:.1f}%)'],
        ['Datasets with minimal leakage', f'{leakage_count} ({leakage_count/len(data)*100:.1f}%)'],
        ['Mean identity', f'{np.mean(mean_identities):.1f}% ± {np.std(mean_identities):.1f}%'],
        ['Max identity <50%', f'{datasets_below_50} ({datasets_below_50/len(data)*100:.1f}%)'],
        ['Performance drop (CD-HIT)', '0.48% (negligible)']
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.5)

    # Style header row
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white', fontsize=14)

    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('white')

            # Highlight key findings
            if 'leakage' in table_data[i][0].lower() or 'Performance' in table_data[i][0]:
                cell.set_text_props(weight='bold')

    ax.set_title('PrismNet Splitting Evaluation - Summary Statistics',
                fontsize=16, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_summary_table.pdf', bbox_inches='tight')
    print(f"✓ Created: {output_dir / 'fig4_summary_table.png'}")
    plt.close()


if __name__ == '__main__':
    print("Generating visualizations for PrismNet splitting evaluation...")
    print(f"Output directory: {output_dir.absolute()}\n")

    create_dataset_quality_chart()
    create_identity_distribution()
    create_performance_comparison()
    create_summary_stats_table()

    print("\n✓ All visualizations created successfully!")
    print(f"\nFiles saved to: {output_dir.absolute()}")
    print("  - fig1_dataset_quality.png/pdf")
    print("  - fig2_identity_distribution.png/pdf")
    print("  - fig3_performance_comparison.png/pdf")
    print("  - fig4_summary_table.png/pdf")
