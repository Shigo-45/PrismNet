"""Homology-aware k-fold cross-validation."""

import numpy as np
import tempfile
from pathlib import Path
from typing import Iterator, List, Tuple
from .cdhit import cluster_sequences
from .analyzer import save_sequences_to_fasta


def homology_aware_kfold(
    sequences: List[Tuple[str, str]],
    n_folds: int = 5,
    method: str = "cdhit",
    identity: float = 0.8,
    random_state: int = 42
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generator yielding (train_idx, test_idx) with no homology across folds.

    Clusters sequences first, then assigns entire clusters to folds ensuring
    no cluster spans multiple folds. This prevents information leakage between
    train and validation sets.

    Args:
        sequences: List of (seq_id, sequence_string) tuples
        n_folds: Number of folds (default: 5)
        method: Clustering method ('cdhit' or 'datasail')
        identity: Sequence identity threshold for clustering
        random_state: Random seed for reproducibility

    Yields:
        Tuples of (train_indices, test_indices) for each fold

    Example:
        >>> sequences = [("seq1", "ACGT"), ("seq2", "ACGG"), ("seq3", "TTTT")]
        >>> for train_idx, test_idx in homology_aware_kfold(sequences, n_folds=3):
        ...     print(f"Train: {train_idx}, Test: {test_idx}")
    """
    rng = np.random.default_rng(random_state)

    # Step 1: Create temporary FASTA file for clustering
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_fasta:
        fasta_path = Path(tmp_fasta.name)
        save_sequences_to_fasta(sequences, fasta_path)

    try:
        # Step 2: Cluster sequences
        if method == "cdhit":
            seq_to_cluster = cluster_sequences(fasta_path, identity=identity)
        elif method == "datasail":
            # For DataSAIL, we use CD-HIT for k-fold since DataSAIL is designed for 2-way splits
            # TODO: Implement DataSAIL-based k-fold if needed
            print("Warning: DataSAIL not supported for k-fold, using CD-HIT instead")
            seq_to_cluster = cluster_sequences(fasta_path, identity=identity)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Clean up temp file
        fasta_path.unlink()

    except Exception as e:
        fasta_path.unlink(missing_ok=True)
        raise e

    # Step 3: Group sequences by cluster
    cluster_to_indices = {}

    for idx, (seq_id, _) in enumerate(sequences):
        if seq_id not in seq_to_cluster:
            # If sequence not in cluster mapping, assign to singleton cluster
            cluster_id = f"singleton_{idx}"
        else:
            cluster_id = seq_to_cluster[seq_id]

        if cluster_id not in cluster_to_indices:
            cluster_to_indices[cluster_id] = []

        cluster_to_indices[cluster_id].append(idx)

    # Step 4: Assign clusters to folds
    cluster_ids = list(cluster_to_indices.keys())
    cluster_sizes = np.array([len(cluster_to_indices[cid]) for cid in cluster_ids])

    # Shuffle clusters
    shuffle_idx = rng.permutation(len(cluster_ids))
    cluster_ids = [cluster_ids[i] for i in shuffle_idx]
    cluster_sizes = cluster_sizes[shuffle_idx]

    # Assign clusters to folds using greedy algorithm to balance fold sizes
    fold_assignments = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)

    for cluster_id, cluster_size in zip(cluster_ids, cluster_sizes):
        # Assign to smallest fold
        smallest_fold = np.argmin(fold_sizes)
        fold_assignments[smallest_fold].append(cluster_id)
        fold_sizes[smallest_fold] += cluster_size

    # Print fold statistics
    total_seqs = sum(cluster_sizes)
    print(f"\nK-fold CV setup ({n_folds} folds):")
    print(f"  Total sequences: {total_seqs}")
    print(f"  Total clusters: {len(cluster_ids)}")
    print(f"  Average fold size: {total_seqs // n_folds}")

    for fold_idx, fold_clusters in enumerate(fold_assignments):
        fold_seq_count = sum(len(cluster_to_indices[cid]) for cid in fold_clusters)
        print(f"  Fold {fold_idx}: {fold_seq_count} sequences ({len(fold_clusters)} clusters)")

    # Step 5: Yield train/test splits for each fold
    for test_fold_idx in range(n_folds):
        # Test set: current fold
        test_clusters = fold_assignments[test_fold_idx]
        test_indices = []
        for cluster_id in test_clusters:
            test_indices.extend(cluster_to_indices[cluster_id])

        # Train set: all other folds
        train_indices = []
        for train_fold_idx in range(n_folds):
            if train_fold_idx != test_fold_idx:
                train_clusters = fold_assignments[train_fold_idx]
                for cluster_id in train_clusters:
                    train_indices.extend(cluster_to_indices[cluster_id])

        # Convert to numpy arrays and sort
        train_indices = np.array(sorted(train_indices))
        test_indices = np.array(sorted(test_indices))

        yield train_indices, test_indices


def stratified_homology_aware_kfold(
    sequences: List[Tuple[str, str]],
    labels: np.ndarray,
    n_folds: int = 5,
    method: str = "cdhit",
    identity: float = 0.8,
    random_state: int = 42
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified k-fold CV that maintains class balance while ensuring no homology across folds.

    This is more complex as it needs to balance both cluster assignment and class distribution.

    Args:
        sequences: List of (seq_id, sequence_string) tuples
        labels: Array of labels for stratification
        n_folds: Number of folds
        method: Clustering method ('cdhit' or 'datasail')
        identity: Sequence identity threshold
        random_state: Random seed

    Yields:
        Tuples of (train_indices, test_indices) for each fold

    Note:
        This is a best-effort stratification. Perfect stratification may not be possible
        when clusters have mixed labels.
    """
    rng = np.random.default_rng(random_state)  # noqa: F841

    # Step 1: Create temporary FASTA file for clustering
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_fasta:
        fasta_path = Path(tmp_fasta.name)
        save_sequences_to_fasta(sequences, fasta_path)

    try:
        # Step 2: Cluster sequences
        if method == "cdhit":
            seq_to_cluster = cluster_sequences(fasta_path, identity=identity)
        else:
            print("Warning: Only CD-HIT supported for stratified k-fold")
            seq_to_cluster = cluster_sequences(fasta_path, identity=identity)

        fasta_path.unlink()

    except Exception as e:
        fasta_path.unlink(missing_ok=True)
        raise e

    # Step 3: Group sequences by cluster and compute cluster label distributions
    cluster_to_indices = {}
    cluster_to_labels = {}

    for idx, (seq_id, _) in enumerate(sequences):
        if seq_id not in seq_to_cluster:
            cluster_id = f"singleton_{idx}"
        else:
            cluster_id = seq_to_cluster[seq_id]

        if cluster_id not in cluster_to_indices:
            cluster_to_indices[cluster_id] = []
            cluster_to_labels[cluster_id] = []

        cluster_to_indices[cluster_id].append(idx)
        cluster_to_labels[cluster_id].append(labels[idx])

    # Step 4: Compute cluster properties
    cluster_info = []

    for cluster_id in cluster_to_indices.keys():
        cluster_labels = np.array(cluster_to_labels[cluster_id])
        cluster_size = len(cluster_labels)

        # Majority class for this cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        majority_class = unique_labels[np.argmax(counts)]
        class_purity = np.max(counts) / cluster_size

        cluster_info.append({
            "id": cluster_id,
            "size": cluster_size,
            "majority_class": majority_class,
            "purity": class_purity,
            "label_dist": dict(zip(unique_labels.tolist(), counts.tolist()))
        })

    # Step 5: Sort clusters by majority class
    cluster_info.sort(key=lambda x: (x["majority_class"], -x["size"]))

    # Step 6: Assign clusters to folds, attempting to balance classes
    unique_labels = sorted(np.unique(labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    fold_assignments = [[] for _ in range(n_folds)]
    fold_label_counts = [np.zeros(len(unique_labels), dtype=int) for _ in range(n_folds)]

    for cluster in cluster_info:
        cluster_id = cluster["id"]

        # Find fold with smallest count of this cluster's majority class
        majority_class_idx = label_to_idx[cluster["majority_class"]]
        fold_majority_counts = [fold_counts[majority_class_idx] for fold_counts in fold_label_counts]
        target_fold = np.argmin(fold_majority_counts)

        # Assign cluster to target fold
        fold_assignments[target_fold].append(cluster_id)

        # Update fold label counts
        for label, count in cluster["label_dist"].items():
            fold_label_counts[target_fold][label_to_idx[label]] += count

    # Print fold statistics
    print(f"\nStratified k-fold CV setup ({n_folds} folds):")

    for fold_idx in range(n_folds):
        fold_counts = fold_label_counts[fold_idx]
        fold_total = sum(fold_counts)
        print(f"  Fold {fold_idx}: {fold_total} sequences", end="")

        for label_idx, label in enumerate(unique_labels):
            count = fold_counts[label_idx]
            pct = count / fold_total * 100 if fold_total > 0 else 0
            print(f" | Class {label}: {count} ({pct:.1f}%)", end="")

        print()

    # Step 7: Yield train/test splits
    for test_fold_idx in range(n_folds):
        test_clusters = fold_assignments[test_fold_idx]
        test_indices = []
        for cluster_id in test_clusters:
            test_indices.extend(cluster_to_indices[cluster_id])

        train_indices = []
        for train_fold_idx in range(n_folds):
            if train_fold_idx != test_fold_idx:
                for cluster_id in fold_assignments[train_fold_idx]:
                    train_indices.extend(cluster_to_indices[cluster_id])

        train_indices = np.array(sorted(train_indices))
        test_indices = np.array(sorted(test_indices))

        yield train_indices, test_indices
