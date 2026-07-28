#!/usr/bin/env python3
"""
Evaluate cross-cell-line prediction performance.

Computes AUC-ROC, AUC-PR, and classification metrics for predictions
made by a model trained on one cell type and tested on another.

Usage:
    python tools/evaluate_cross_cell.py --probs <probs_file> --tsv <tsv_file> [--output <output_file>]

Example:
    python tools/evaluate_cross_cell.py \
        --probs exp/prismnet/out/infer/IGF2BP1_K562_PrismNet_pu_IGF2BP1_HepG2.tsv.probs \
        --tsv data/clip_data/IGF2BP1_HepG2.tsv \
        --output exp/prismnet/out/IGF2BP1_cross_cell_eval.txt
"""

import argparse
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)


def load_predictions(probs_file):
    """Load prediction probabilities from file."""
    return np.loadtxt(probs_file)


def load_labels(tsv_file):
    """Load ground truth labels from TSV file."""
    labels = []
    with open(tsv_file, 'r') as f:
        header = next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            labels.append(int(parts[5]))  # label column
    return np.array(labels)


def evaluate(labels, probs, threshold=0.5):
    """Compute evaluation metrics."""
    results = {}

    # Dataset statistics
    results['n_samples'] = len(labels)
    results['n_positives'] = int(sum(labels))
    results['n_negatives'] = int(sum(1 - labels))
    results['pos_ratio'] = sum(labels) / len(labels)

    # AUC-ROC
    results['auroc'] = roc_auc_score(labels, probs)

    # AUC-PR
    precision, recall, _ = precision_recall_curve(labels, probs)
    results['auprc'] = auc(recall, precision)

    # ROC curve points (for plotting)
    fpr, tpr, _ = roc_curve(labels, probs)
    results['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
    results['pr_curve'] = {'precision': precision, 'recall': recall}

    # Binary predictions at threshold
    preds = (probs >= threshold).astype(int)
    results['threshold'] = threshold
    results['accuracy'] = accuracy_score(labels, preds)
    results['precision'] = precision_score(labels, preds, zero_division=0)
    results['recall'] = recall_score(labels, preds, zero_division=0)
    results['f1'] = f1_score(labels, preds, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    results['true_positives'] = int(tp)
    results['true_negatives'] = int(tn)
    results['false_positives'] = int(fp)
    results['false_negatives'] = int(fn)
    results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Prediction distribution
    results['n_pred_positive'] = int(sum(preds))
    results['mean_prob_true_pos'] = float(probs[labels == 1].mean()) if sum(labels) > 0 else 0
    results['mean_prob_true_neg'] = float(probs[labels == 0].mean()) if sum(1 - labels) > 0 else 0

    return results


def print_results(results, train_name="K562", test_name="HepG2"):
    """Print formatted evaluation results."""
    print("=" * 60)
    print("Cross-Cell-Line Prediction Results")
    print(f"Train: {train_name} | Test: {test_name}")
    print("=" * 60)

    print(f"\nDataset:")
    print(f"  Total samples: {results['n_samples']}")
    print(f"  Positives: {results['n_positives']} ({100*results['pos_ratio']:.1f}%)")
    print(f"  Negatives: {results['n_negatives']} ({100*(1-results['pos_ratio']):.1f}%)")

    print(f"\nPrimary Metrics:")
    print(f"  AUC-ROC: {results['auroc']:.4f}")
    print(f"  AUC-PR:  {results['auprc']:.4f}")

    print(f"\nAt threshold {results['threshold']}:")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Precision:   {results['precision']:.4f}")
    print(f"  Recall:      {results['recall']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")
    print(f"  F1-score:    {results['f1']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {results['true_positives']}")
    print(f"  True Negatives:  {results['true_negatives']}")
    print(f"  False Positives: {results['false_positives']}")
    print(f"  False Negatives: {results['false_negatives']}")

    print(f"\nPrediction Distribution:")
    print(f"  Predicted positive: {results['n_pred_positive']}")
    print(f"  Mean prob (true positives): {results['mean_prob_true_pos']:.4f}")
    print(f"  Mean prob (true negatives): {results['mean_prob_true_neg']:.4f}")


def save_results(results, output_file, train_name="K562", test_name="HepG2"):
    """Save results to file."""
    with open(output_file, 'w') as f:
        f.write("Cross-Cell-Line Prediction Evaluation\n")
        f.write(f"Train: {train_name} | Test: {test_name}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Dataset:\n")
        f.write(f"  Total samples: {results['n_samples']}\n")
        f.write(f"  Positives: {results['n_positives']} ({100*results['pos_ratio']:.1f}%)\n")
        f.write(f"  Negatives: {results['n_negatives']} ({100*(1-results['pos_ratio']):.1f}%)\n\n")

        f.write(f"Primary Metrics:\n")
        f.write(f"  AUC-ROC: {results['auroc']:.4f}\n")
        f.write(f"  AUC-PR:  {results['auprc']:.4f}\n\n")

        f.write(f"Classification Metrics (threshold={results['threshold']}):\n")
        f.write(f"  Accuracy:    {results['accuracy']:.4f}\n")
        f.write(f"  Precision:   {results['precision']:.4f}\n")
        f.write(f"  Recall:      {results['recall']:.4f}\n")
        f.write(f"  Specificity: {results['specificity']:.4f}\n")
        f.write(f"  F1-score:    {results['f1']:.4f}\n\n")

        f.write(f"Confusion Matrix:\n")
        f.write(f"  TP={results['true_positives']} FP={results['false_positives']}\n")
        f.write(f"  FN={results['false_negatives']} TN={results['true_negatives']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cross-cell-line prediction performance"
    )
    parser.add_argument(
        "--probs", required=True,
        help="Path to prediction probabilities file (.probs)"
    )
    parser.add_argument(
        "--tsv", required=True,
        help="Path to TSV file with ground truth labels"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file for results (optional)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold (default: 0.5)"
    )
    parser.add_argument(
        "--train-name", default="K562",
        help="Name of training cell type (default: K562)"
    )
    parser.add_argument(
        "--test-name", default="HepG2",
        help="Name of testing cell type (default: HepG2)"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading predictions from: {args.probs}")
    probs = load_predictions(args.probs)

    print(f"Loading labels from: {args.tsv}")
    labels = load_labels(args.tsv)

    # Validate
    if len(probs) != len(labels):
        raise ValueError(f"Mismatch: {len(probs)} predictions vs {len(labels)} labels")

    # Evaluate
    results = evaluate(labels, probs, threshold=args.threshold)

    # Print results
    print_results(results, train_name=args.train_name, test_name=args.test_name)

    # Save if output specified
    if args.output:
        save_results(results, args.output, train_name=args.train_name, test_name=args.test_name)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
