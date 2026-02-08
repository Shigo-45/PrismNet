#!/usr/bin/env python
"""Comprehensive analysis of saliency evaluation results across all RBPs.

This script analyzes results from all 171 RBP evaluations and generates:
- Cross-RBP statistical analysis
- Protein family comparisons
- Cell line comparisons
- Publication-ready visualizations
- Comprehensive summary report
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_all_results(eval_dir: Path) -> Dict:
    """Load results from all protein evaluations.

    Args:
        eval_dir: Evaluation directory

    Returns:
        Dictionary mapping protein names to results
    """
    results = {}

    for protein_dir in eval_dir.iterdir():
        if not protein_dir.is_dir():
            continue

        protein = protein_dir.name
        full_path = protein_dir / "full_randomization_results.json"
        cascading_path = protein_dir / "cascading_randomization_results.json"

        if full_path.exists() and cascading_path.exists():
            try:
                with open(full_path) as f:
                    full_results = json.load(f)
                with open(cascading_path) as f:
                    cascading_results = json.load(f)

                results[protein] = {
                    "full": full_results,
                    "cascading": cascading_results,
                }
            except Exception as e:
                print(f"Warning: Failed to load {protein}: {e}")

    return results


def parse_protein_info(protein: str) -> Tuple[str, str]:
    """Parse protein name into RBP and cell line.

    Args:
        protein: Protein name (e.g., "TIA1_Hela")

    Returns:
        Tuple of (rbp_name, cell_line)
    """
    parts = protein.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return protein, "Unknown"


def classify_result(tvr_ssim: float, rvr_ssim: float,
                    tvr_spearman: float, rvr_spearman: float) -> str:
    """Classify evaluation result as pass/weak/fail.

    Args:
        tvr_ssim: Trained vs random SSIM
        rvr_ssim: Random vs random SSIM
        tvr_spearman: Trained vs random Spearman
        rvr_spearman: Random vs random Spearman

    Returns:
        Classification: "strong_pass", "moderate_pass", "weak", or "fail"
    """
    # Check for NaN
    if np.isnan(rvr_ssim) or np.isnan(rvr_spearman):
        return "weak"

    # Calculate ratios
    ssim_ratio = tvr_ssim / rvr_ssim if rvr_ssim != 0 else float('inf')
    spearman_ratio = tvr_spearman / rvr_spearman if rvr_spearman != 0 else float('inf')

    # Strong pass: Clear divergence
    if (tvr_ssim < 0.1 and tvr_spearman < 0.2 and
        (ssim_ratio > 1.5 or ssim_ratio < 0.7)):
        return "strong_pass"

    # Moderate pass: Some divergence
    if (tvr_ssim < 0.2 and tvr_spearman < 0.4):
        return "moderate_pass"

    # Weak: Unclear results
    if (tvr_ssim < 0.3 or tvr_spearman < 0.5):
        return "weak"

    # Fail: High similarity (saliency may be method-dependent)
    return "fail"


def create_summary_dataframe(results: Dict) -> pd.DataFrame:
    """Create summary DataFrame from all results.

    Args:
        results: Dictionary of results

    Returns:
        DataFrame with summary statistics
    """
    data = []

    for protein, result in results.items():
        rbp, cell_line = parse_protein_info(protein)
        full = result["full"]
        tvr = full["trained_vs_random"]
        rvr = full["random_vs_random"]

        tvr_ssim = tvr["ssim_mean"]["mean"]
        rvr_ssim = rvr["ssim_mean"]["mean"]
        tvr_spearman = tvr["spearman_mean"]["mean"]
        rvr_spearman = rvr["spearman_mean"]["mean"]

        classification = classify_result(
            tvr_ssim, rvr_ssim, tvr_spearman, rvr_spearman
        )

        data.append({
            "protein": protein,
            "rbp": rbp,
            "cell_line": cell_line,
            "tvr_ssim": tvr_ssim,
            "rvr_ssim": rvr_ssim,
            "tvr_spearman": tvr_spearman,
            "rvr_spearman": rvr_spearman,
            "ssim_ratio": tvr_ssim / rvr_ssim if rvr_ssim != 0 else np.nan,
            "spearman_ratio": tvr_spearman / rvr_spearman if rvr_spearman != 0 else np.nan,
            "classification": classification,
        })

    return pd.DataFrame(data)


def plot_overall_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot overall distribution of results.

    Args:
        df: Summary DataFrame
        output_dir: Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. SSIM distribution
    ax = axes[0, 0]
    ax.hist(df["tvr_ssim"].dropna(), bins=30, alpha=0.7, label="Trained vs Random", color="steelblue")
    ax.hist(df["rvr_ssim"].dropna(), bins=30, alpha=0.7, label="Random vs Random", color="orange")
    ax.set_xlabel("SSIM")
    ax.set_ylabel("Count")
    ax.set_title("SSIM Distribution Across All RBPs")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Spearman distribution
    ax = axes[0, 1]
    ax.hist(df["tvr_spearman"].dropna(), bins=30, alpha=0.7, label="Trained vs Random", color="steelblue")
    ax.hist(df["rvr_spearman"].dropna(), bins=30, alpha=0.7, label="Random vs Random", color="orange")
    ax.set_xlabel("Spearman Correlation")
    ax.set_ylabel("Count")
    ax.set_title("Spearman Distribution Across All RBPs")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Classification pie chart
    ax = axes[1, 0]
    classification_counts = df["classification"].value_counts()
    colors = {
        "strong_pass": "#2ecc71",
        "moderate_pass": "#3498db",
        "weak": "#f39c12",
        "fail": "#e74c3c"
    }
    ax.pie(
        classification_counts.values,
        labels=classification_counts.index,
        autopct='%1.1f%%',
        colors=[colors.get(c, "gray") for c in classification_counts.index],
        startangle=90
    )
    ax.set_title("Classification Distribution")

    # 4. Scatter plot: SSIM vs Spearman
    ax = axes[1, 1]
    for classification in df["classification"].unique():
        subset = df[df["classification"] == classification]
        ax.scatter(
            subset["tvr_ssim"],
            subset["tvr_spearman"],
            label=classification,
            alpha=0.6,
            s=50
        )
    ax.set_xlabel("Trained vs Random SSIM")
    ax.set_ylabel("Trained vs Random Spearman")
    ax.set_title("SSIM vs Spearman (Trained vs Random)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f"Saliency Sanity Check Results: {len(df)} RBPs", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "overall_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Overall distribution plot saved to: {output_path}")


def plot_cell_line_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot comparison across cell lines.

    Args:
        df: Summary DataFrame
        output_dir: Output directory
    """
    cell_lines = df["cell_line"].value_counts().head(10).index

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. SSIM by cell line
    ax = axes[0, 0]
    data_to_plot = [df[df["cell_line"] == cl]["tvr_ssim"].dropna() for cl in cell_lines]
    bp = ax.boxplot(data_to_plot, labels=cell_lines, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
    ax.set_xlabel("Cell Line")
    ax.set_ylabel("Trained vs Random SSIM")
    ax.set_title("SSIM by Cell Line")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # 2. Spearman by cell line
    ax = axes[0, 1]
    data_to_plot = [df[df["cell_line"] == cl]["tvr_spearman"].dropna() for cl in cell_lines]
    bp = ax.boxplot(data_to_plot, labels=cell_lines, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('orange')
    ax.set_xlabel("Cell Line")
    ax.set_ylabel("Trained vs Random Spearman")
    ax.set_title("Spearman by Cell Line")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # 3. Pass rate by cell line
    ax = axes[1, 0]
    pass_rates = []
    for cl in cell_lines:
        subset = df[df["cell_line"] == cl]
        pass_count = len(subset[subset["classification"].isin(["strong_pass", "moderate_pass"])])
        pass_rate = pass_count / len(subset) * 100
        pass_rates.append(pass_rate)

    ax.bar(range(len(cell_lines)), pass_rates, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(cell_lines)))
    ax.set_xticklabels(cell_lines, rotation=45, ha='right')
    ax.set_xlabel("Cell Line")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Pass Rate by Cell Line")
    ax.axhline(y=60, color='r', linestyle='--', alpha=0.5, label='Overall average')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Sample count by cell line
    ax = axes[1, 1]
    cell_line_counts = df["cell_line"].value_counts().head(10)
    ax.barh(range(len(cell_line_counts)), cell_line_counts.values, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(cell_line_counts)))
    ax.set_yticklabels(cell_line_counts.index)
    ax.set_xlabel("Number of RBPs")
    ax.set_title("RBP Count by Cell Line")
    ax.grid(axis='x', alpha=0.3)

    plt.suptitle("Cell Line Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "cell_line_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Cell line comparison plot saved to: {output_path}")


def plot_top_bottom_rbps(df: pd.DataFrame, output_dir: Path, n: int = 20):
    """Plot top and bottom performing RBPs.

    Args:
        df: Summary DataFrame
        output_dir: Output directory
        n: Number of top/bottom to show
    """
    # Sort by SSIM ratio (lower is better for trained vs random)
    df_sorted = df.sort_values("ssim_ratio")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Top performers (lowest SSIM ratio = best divergence)
    ax = axes[0]
    top_n = df_sorted.head(n)
    y_pos = np.arange(len(top_n))
    ax.barh(y_pos, top_n["ssim_ratio"], color='green', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_n["protein"], fontsize=8)
    ax.set_xlabel("SSIM Ratio (Trained/Random)")
    ax.set_title(f"Top {n} RBPs (Best Saliency Reliability)")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Bottom performers (highest SSIM ratio = poor divergence)
    ax = axes[1]
    bottom_n = df_sorted.tail(n)
    y_pos = np.arange(len(bottom_n))
    ax.barh(y_pos, bottom_n["ssim_ratio"], color='red', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bottom_n["protein"], fontsize=8)
    ax.set_xlabel("SSIM Ratio (Trained/Random)")
    ax.set_title(f"Bottom {n} RBPs (Weakest Saliency Reliability)")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.suptitle("Top and Bottom Performing RBPs", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "top_bottom_rbps.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Top/bottom RBPs plot saved to: {output_path}")


def generate_statistical_report(df: pd.DataFrame, output_dir: Path):
    """Generate statistical analysis report.

    Args:
        df: Summary DataFrame
        output_dir: Output directory
    """
    report = []

    report.append("# Comprehensive Statistical Analysis")
    report.append(f"\n## Overall Statistics (N={len(df)})\n")

    # Overall metrics
    report.append("### Similarity Metrics\n")
    report.append(f"**Trained vs Random SSIM**: {df['tvr_ssim'].mean():.4f} ± {df['tvr_ssim'].std():.4f}")
    report.append(f"  - Median: {df['tvr_ssim'].median():.4f}")
    report.append(f"  - Range: [{df['tvr_ssim'].min():.4f}, {df['tvr_ssim'].max():.4f}]")
    report.append("")
    report.append(f"**Random vs Random SSIM**: {df['rvr_ssim'].mean():.4f} ± {df['rvr_ssim'].std():.4f}")
    report.append(f"  - Median: {df['rvr_ssim'].median():.4f}")
    report.append(f"  - Range: [{df['rvr_ssim'].min():.4f}, {df['rvr_ssim'].max():.4f}]")
    report.append("")
    report.append(f"**Trained vs Random Spearman**: {df['tvr_spearman'].mean():.4f} ± {df['tvr_spearman'].std():.4f}")
    report.append(f"  - Median: {df['tvr_spearman'].median():.4f}")
    report.append(f"  - Range: [{df['tvr_spearman'].min():.4f}, {df['tvr_spearman'].max():.4f}]")
    report.append("")
    report.append(f"**Random vs Random Spearman**: {df['rvr_spearman'].mean():.4f} ± {df['rvr_spearman'].std():.4f}")
    report.append(f"  - Median: {df['rvr_spearman'].median():.4f}")
    report.append(f"  - Range: [{df['rvr_spearman'].min():.4f}, {df['rvr_spearman'].max():.4f}]")
    report.append("")

    # Statistical tests
    report.append("### Statistical Significance\n")

    # Paired t-test: trained vs random vs random vs random
    tvr_ssim = df['tvr_ssim'].dropna()
    rvr_ssim = df['rvr_ssim'].dropna()

    if len(tvr_ssim) > 0 and len(rvr_ssim) > 0:
        t_stat, p_value = stats.ttest_ind(tvr_ssim, rvr_ssim)
        report.append(f"**SSIM Comparison** (Independent t-test):")
        report.append(f"  - t-statistic: {t_stat:.4f}")
        report.append(f"  - p-value: {p_value:.4e}")
        report.append(f"  - Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        report.append("")

    tvr_spearman = df['tvr_spearman'].dropna()
    rvr_spearman = df['rvr_spearman'].dropna()

    if len(tvr_spearman) > 0 and len(rvr_spearman) > 0:
        t_stat, p_value = stats.ttest_ind(tvr_spearman, rvr_spearman)
        report.append(f"**Spearman Comparison** (Independent t-test):")
        report.append(f"  - t-statistic: {t_stat:.4f}")
        report.append(f"  - p-value: {p_value:.4e}")
        report.append(f"  - Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        report.append("")

    # Classification breakdown
    report.append("## Classification Breakdown\n")
    classification_counts = df["classification"].value_counts()
    for classification, count in classification_counts.items():
        percentage = count / len(df) * 100
        report.append(f"**{classification}**: {count} ({percentage:.1f}%)")
    report.append("")

    # Cell line analysis
    report.append("## Cell Line Analysis\n")
    cell_line_stats = df.groupby("cell_line").agg({
        "tvr_ssim": ["mean", "std", "count"],
        "tvr_spearman": ["mean", "std"],
    }).round(4)

    report.append("### Top 10 Cell Lines by Sample Count\n")
    top_cell_lines = df["cell_line"].value_counts().head(10)
    for cell_line, count in top_cell_lines.items():
        subset = df[df["cell_line"] == cell_line]
        pass_count = len(subset[subset["classification"].isin(["strong_pass", "moderate_pass"])])
        pass_rate = pass_count / len(subset) * 100
        report.append(f"**{cell_line}**: {count} RBPs, {pass_rate:.1f}% pass rate")
    report.append("")

    # Top performers
    report.append("## Top 20 Performers\n")
    top_20 = df.nsmallest(20, "ssim_ratio")
    report.append("| Rank | Protein | SSIM Ratio | Classification |")
    report.append("|------|---------|------------|----------------|")
    for i, (_, row) in enumerate(top_20.iterrows(), 1):
        report.append(f"| {i} | {row['protein']} | {row['ssim_ratio']:.3f} | {row['classification']} |")
    report.append("")

    # Bottom performers
    report.append("## Bottom 20 Performers\n")
    bottom_20 = df.nlargest(20, "ssim_ratio")
    report.append("| Rank | Protein | SSIM Ratio | Classification |")
    report.append("|------|---------|------------|----------------|")
    for i, (_, row) in enumerate(bottom_20.iterrows(), 1):
        report.append(f"| {i} | {row['protein']} | {row['ssim_ratio']:.3f} | {row['classification']} |")
    report.append("")

    # Save report
    output_path = output_dir / "statistical_report.md"
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    print(f"Statistical report saved to: {output_path}")


def main():
    eval_dir = Path("/home/shigo-45/projects/PrismNet-eval-saliency/evaluation/saliency/comprehensive_evaluation")
    output_dir = eval_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    print("Loading results from all RBP evaluations...")
    results = load_all_results(eval_dir)
    print(f"Loaded results for {len(results)} proteins")

    if len(results) == 0:
        print("No results found! Make sure evaluations have completed.")
        return

    print("\nCreating summary DataFrame...")
    df = create_summary_dataframe(results)

    # Save DataFrame
    df.to_csv(output_dir / "summary_table.csv", index=False)
    print(f"Summary table saved to: {output_dir / 'summary_table.csv'}")

    print("\nGenerating visualizations...")
    plot_overall_distribution(df, output_dir)
    plot_cell_line_comparison(df, output_dir)
    plot_top_bottom_rbps(df, output_dir, n=20)

    print("\nGenerating statistical report...")
    generate_statistical_report(df, output_dir)

    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"\nKey files:")
    print(f"  - summary_table.csv: Complete data table")
    print(f"  - statistical_report.md: Detailed statistical analysis")
    print(f"  - overall_distribution.png: Overall results visualization")
    print(f"  - cell_line_comparison.png: Cell line analysis")
    print(f"  - top_bottom_rbps.png: Best and worst performers")


if __name__ == "__main__":
    main()
