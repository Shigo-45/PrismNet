"""CD-HIT wrapper for sequence clustering and homology-aware splitting."""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def cluster_sequences(fasta_path: Path, identity: float = 0.8) -> Dict[str, int]:
    """
    Run CD-HIT and return sequence->cluster mapping.

    Args:
        fasta_path: Path to input FASTA file
        identity: Sequence identity threshold (0-1)

    Returns:
        Dictionary mapping sequence IDs to cluster IDs

    Raises:
        FileNotFoundError: If cd-hit binary not found
        RuntimeError: If cd-hit execution fails
    """
    fasta_path = Path(fasta_path)

    # Check if cd-hit is available
    if not shutil.which("cd-hit"):
        raise FileNotFoundError(
            "cd-hit not found. Install with: conda install -c bioconda cd-hit"
        )

    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as tmp_out:
        output_path = Path(tmp_out.name)

    try:
        # Run cd-hit
        # -c: sequence identity threshold
        # -n: word length (5 recommended for 0.7-1.0 identity)
        # -M: memory limit in MB (0 = unlimited)
        # -T: number of threads
        cmd = [
            "cd-hit",
            "-i", str(fasta_path),
            "-o", str(output_path),
            "-c", str(identity),
            "-n", "5",
            "-M", "0",
            "-T", "0",  # Use all available threads
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse cluster file
        cluster_file = Path(str(output_path) + ".clstr")
        seq_to_cluster = _parse_clstr_file(cluster_file)

        # Clean up temporary files
        output_path.unlink(missing_ok=True)
        cluster_file.unlink(missing_ok=True)

        return seq_to_cluster

    except subprocess.CalledProcessError as e:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"cd-hit failed: {e.stderr}")


def _parse_clstr_file(clstr_path: Path) -> Dict[str, int]:
    """
    Parse CD-HIT .clstr output file.

    Format example:
        >Cluster 0
        0	100aa, >seq1... *
        1	98aa, >seq2... at 95.00%
        >Cluster 1
        0	102aa, >seq3... *

    Returns:
        Dictionary mapping sequence IDs to cluster IDs
    """
    seq_to_cluster = {}
    current_cluster = -1

    with open(clstr_path) as f:
        for line in f:
            line = line.strip()

            if line.startswith(">Cluster"):
                # New cluster
                current_cluster = int(line.split()[1])
            elif line:
                # Sequence line - extract ID
                # Format: "0	100aa, >seq_id... at 95.00%" or "... *"
                parts = line.split(">")
                if len(parts) > 1:
                    # Extract sequence ID (up to first space, dot, or ...)
                    seq_id = parts[1].split("...")[0].split()[0].rstrip(".")
                    seq_to_cluster[seq_id] = current_cluster

    assert len(seq_to_cluster) > 0, "No sequences found in cluster file"
    return seq_to_cluster


def split_by_clusters(
    clusters: Dict[str, int],
    test_fraction: float = 0.2,
    random_state: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Assign entire clusters to train or test sets.

    Uses stratified splitting to maintain balance between sets.

    Args:
        clusters: Dictionary mapping sequence IDs to cluster IDs
        test_fraction: Fraction of sequences for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_seq_ids, test_seq_ids)
    """
    np.random.seed(random_state)

    # Group sequences by cluster
    cluster_to_seqs = {}
    for seq_id, cluster_id in clusters.items():
        if cluster_id not in cluster_to_seqs:
            cluster_to_seqs[cluster_id] = []
        cluster_to_seqs[cluster_id].append(seq_id)

    # Get cluster IDs and sizes
    cluster_ids = list(cluster_to_seqs.keys())
    cluster_sizes = np.array([len(cluster_to_seqs[cid]) for cid in cluster_ids])

    # Shuffle clusters
    shuffle_idx = np.random.permutation(len(cluster_ids))
    cluster_ids = [cluster_ids[i] for i in shuffle_idx]
    cluster_sizes = cluster_sizes[shuffle_idx]

    # Greedily assign clusters to test set until we reach target fraction
    total_seqs = sum(cluster_sizes)
    target_test_size = int(total_seqs * test_fraction)

    test_cluster_ids = []
    test_size = 0

    for cid, size in zip(cluster_ids, cluster_sizes):
        if test_size < target_test_size:
            test_cluster_ids.append(cid)
            test_size += size
        else:
            break

    # Collect sequence IDs
    train_seqs = []
    test_seqs = []

    for cid in cluster_ids:
        seqs = cluster_to_seqs[cid]
        if cid in test_cluster_ids:
            test_seqs.extend(seqs)
        else:
            train_seqs.extend(seqs)

    print(f"Split: {len(train_seqs)} train, {len(test_seqs)} test "
          f"({len(test_seqs)/total_seqs*100:.1f}% test)")
    print(f"Clusters: {len(cluster_ids) - len(test_cluster_ids)} train, "
          f"{len(test_cluster_ids)} test")

    return train_seqs, test_seqs
