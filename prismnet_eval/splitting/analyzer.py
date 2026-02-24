"""Homology analysis for existing and new data splits."""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from Bio import Align
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


def extract_sequences_from_h5(h5_path: Path, dataset: str = "X_train") -> List[Tuple[str, str]]:
    """
    Extract sequences from H5 file as (id, sequence) tuples.

    Args:
        h5_path: Path to H5 file
        dataset: Dataset name ('X_train', 'X_test', etc.)

    Returns:
        List of (seq_id, sequence_string) tuples
    """
    with h5py.File(h5_path, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset {dataset} not found in {h5_path}")

        X = f[dataset][:]  # Shape: (samples, features, seq_len) or (samples, 1, seq_len, n_features)

    assert X.ndim in (3, 4), f"Unexpected shape: {X.shape}"
    was_4d = X.ndim == 4
    if was_4d:
        X = X[:, 0, :, :]  # Remove channel dim -> (samples, seq_len, n_features)

    # Extract nucleotide sequences from one-hot encoding
    # First 4 features are A, C, G, T
    nucleotides = "ACGT"
    sequences = []

    for i, sample in enumerate(X):
        if not was_4d:
            # 3D: shape (samples, features, seq_len), sample is (features, seq_len)
            one_hot = sample[:4, :]  # Shape: (4, seq_len)
            seq_indices = np.argmax(one_hot, axis=0)
        else:
            # 4D squeezed to 3D: shape (samples, seq_len, n_features), sample is (seq_len, n_features)
            one_hot = sample[:, :4].T  # Shape: (4, seq_len)
            seq_indices = np.argmax(one_hot, axis=0)

        sequence = "".join(nucleotides[idx] for idx in seq_indices)

        # Create unique ID
        seq_id = f"{dataset}_{i}"
        sequences.append((seq_id, sequence))

    return sequences


def save_sequences_to_fasta(sequences: List[Tuple[str, str]], output_path: Path):
    """Save sequences to FASTA file."""
    with open(output_path, "w") as f:
        for seq_id, seq in sequences:
            f.write(f">{seq_id}\n{seq}\n")


def compute_pairwise_identity(seq1: str, seq2: str) -> float:
    """
    Compute sequence identity between two sequences.

    Uses simple position-by-position comparison for equal-length sequences,
    or alignment-based comparison for different lengths.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Identity score (0-1)
    """
    if not HAS_BIOPYTHON:
        raise ImportError("biopython is required. Install with: uv add biopython")

    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0

    # For equal-length sequences, use simple position comparison (faster)
    if len(seq1) == len(seq2):
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return matches / len(seq1)

    # For different lengths, use alignment with overflow protection
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.gap_score = -1  # Penalize gaps to reduce alignment count

    try:
        alignments = aligner.align(seq1, seq2)

        # Get the best alignment (first one)
        alignment = alignments[0]

        # Count matches
        matches = sum(1 for a, b in zip(alignment[0], alignment[1]) if a == b and a != "-" and b != "-")
        total = max(len(seq1), len(seq2))

        return matches / total if total > 0 else 0.0

    except (OverflowError, MemoryError):
        # Fallback: use simple overlap comparison
        min_len = min(len(seq1), len(seq2))
        matches = sum(seq1[i] == seq2[i] for i in range(min_len))
        return matches / max(len(seq1), len(seq2))


def analyze_split_homology(
    train_sequences: List[Tuple[str, str]],
    test_sequences: List[Tuple[str, str]],
    identity_threshold: float = 0.8,
    sample_size: int = None
) -> Dict:
    """
    Analyze homology between train and test sets.

    Args:
        train_sequences: List of (id, sequence) tuples for train set
        test_sequences: List of (id, sequence) tuples for test set
        identity_threshold: Identity threshold for counting leakage
        sample_size: If set, randomly sample this many pairs (for speed)

    Returns:
        Dictionary with leakage statistics
    """
    print(f"Analyzing homology between {len(train_sequences)} train and {len(test_sequences)} test sequences...")

    # Extract just sequences
    train_seqs = [seq for _, seq in train_sequences]
    test_seqs = [seq for _, seq in test_sequences]

    # Sample if requested
    if sample_size and len(train_seqs) * len(test_seqs) > sample_size:
        n_train_sample = min(len(train_seqs), int(np.sqrt(sample_size)))
        n_test_sample = min(len(test_seqs), sample_size // n_train_sample)

        train_indices = np.random.choice(len(train_seqs), n_train_sample, replace=False)
        test_indices = np.random.choice(len(test_seqs), n_test_sample, replace=False)

        train_seqs = [train_seqs[i] for i in train_indices]
        test_seqs = [test_seqs[i] for i in test_indices]

        print(f"Sampling {len(train_seqs)} train x {len(test_seqs)} test pairs")

    # Compute pairwise identities
    identities = []
    pairs_above_threshold = 0

    for i, train_seq in enumerate(train_seqs):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(train_seqs)} train sequences...")

        for test_seq in test_seqs:
            identity = compute_pairwise_identity(train_seq, test_seq)
            identities.append(identity)

            if identity >= identity_threshold:
                pairs_above_threshold += 1

    identities = np.array(identities)

    # Compute statistics
    stats = {
        "n_train": len(train_sequences),
        "n_test": len(test_sequences),
        "n_pairs_analyzed": len(identities),
        "n_pairs_above_threshold": pairs_above_threshold,
        "leakage_fraction": pairs_above_threshold / len(identities) if len(identities) > 0 else 0.0,
        "max_identity": float(np.max(identities)) if len(identities) > 0 else 0.0,
        "mean_identity": float(np.mean(identities)) if len(identities) > 0 else 0.0,
        "median_identity": float(np.median(identities)) if len(identities) > 0 else 0.0,
        "std_identity": float(np.std(identities)) if len(identities) > 0 else 0.0,
        "identity_threshold": identity_threshold,
    }

    # Add percentile information
    if len(identities) > 0:
        stats["percentile_90"] = float(np.percentile(identities, 90))
        stats["percentile_95"] = float(np.percentile(identities, 95))
        stats["percentile_99"] = float(np.percentile(identities, 99))

    return stats


def analyze_existing_split(
    h5_path: Path,
    identity_threshold: float = 0.8,
    sample_size: int = 10000
) -> Dict:
    """
    Quantify homology between train and test in existing split.

    Args:
        h5_path: Path to H5 file with existing split
        identity_threshold: Identity threshold for leakage
        sample_size: Number of pairs to sample (None = all pairs)

    Returns:
        Dictionary with leakage statistics
    """
    h5_path = Path(h5_path)

    # Extract sequences
    train_sequences = extract_sequences_from_h5(h5_path, "X_train")
    test_sequences = extract_sequences_from_h5(h5_path, "X_test")

    # Analyze homology
    stats = analyze_split_homology(
        train_sequences,
        test_sequences,
        identity_threshold=identity_threshold,
        sample_size=sample_size
    )

    stats["split_method"] = "original"

    return stats


def create_homology_report(
    original: Dict,
    cdhit: Dict = None,
    datasail: Dict = None
) -> Dict:
    """
    Compare leakage metrics across splitting methods.

    Args:
        original: Stats from original split
        cdhit: Stats from CD-HIT split (optional)
        datasail: Stats from DataSAIL split (optional)

    Returns:
        Comparison report dictionary
    """
    report = {
        "splits": {},
        "summary": {}
    }

    # Add each split's stats
    report["splits"]["original"] = original

    if cdhit:
        report["splits"]["cdhit"] = cdhit

    if datasail:
        report["splits"]["datasail"] = datasail

    # Create summary comparison
    methods = ["original"]
    if cdhit:
        methods.append("cdhit")
    if datasail:
        methods.append("datasail")

    # Compare key metrics
    summary = {
        "methods": methods,
        "max_identity": {},
        "mean_identity": {},
        "leakage_fraction": {},
        "pairs_above_threshold": {},
    }

    for method in methods:
        stats = report["splits"][method]
        summary["max_identity"][method] = stats["max_identity"]
        summary["mean_identity"][method] = stats["mean_identity"]
        summary["leakage_fraction"][method] = stats["leakage_fraction"]
        summary["pairs_above_threshold"][method] = stats["n_pairs_above_threshold"]

    # Calculate improvements
    if len(methods) > 1:
        orig_leakage = original["leakage_fraction"]

        for method in methods[1:]:
            new_leakage = report["splits"][method]["leakage_fraction"]
            improvement = (orig_leakage - new_leakage) / orig_leakage if orig_leakage > 0 else 0.0
            summary[f"{method}_improvement"] = improvement

    report["summary"] = summary

    return report


def create_fasta_from_h5(h5_path: Path, output_dir: Path = None) -> Path:
    """
    Create FASTA file from H5 dataset for use with CD-HIT/DataSAIL.

    Combines train and test sequences into single FASTA.

    Args:
        h5_path: Path to H5 file
        output_dir: Output directory (defaults to same as h5_path)

    Returns:
        Path to created FASTA file
    """
    h5_path = Path(h5_path)

    if output_dir is None:
        output_dir = h5_path.parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    fasta_path = output_dir / f"{h5_path.stem}.fasta"

    # Extract all sequences
    all_sequences = []

    for dataset in ["X_train", "X_test"]:
        sequences = extract_sequences_from_h5(h5_path, dataset)
        all_sequences.extend(sequences)

    # Save to FASTA
    save_sequences_to_fasta(all_sequences, fasta_path)

    print(f"Created FASTA with {len(all_sequences)} sequences: {fasta_path}")

    return fasta_path
