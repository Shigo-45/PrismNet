"""DataSAIL wrapper for leakage-minimizing data splitting."""

import tempfile
from pathlib import Path
from typing import List, Tuple
import subprocess


def split_with_datasail(
    fasta_path: Path,
    identity: float = 0.8,
    test_fraction: float = 0.2,
    random_state: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Use DataSAIL for optimal leakage-minimizing split.

    DataSAIL (Data Splitting Against Information Leakage) creates train/test
    splits that minimize similarity between sets while maintaining balance.

    Args:
        fasta_path: Path to input FASTA file
        identity: Maximum sequence identity threshold between train/test
        test_fraction: Fraction of sequences for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_seq_ids, test_seq_ids)

    Raises:
        ImportError: If datasail package not installed
        RuntimeError: If datasail execution fails
    """
    try:
        from datasail.sail import datasail
        from datasail.reader.read_molecules import read_fasta
    except ImportError:
        raise ImportError(
            "datasail not found. Install with: pip install datasail"
        )

    fasta_path = Path(fasta_path)

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)

        # Read sequences
        molecules = read_fasta(str(fasta_path))

        # Run DataSAIL
        # DataSAIL parameters:
        # - techniques: ["M"] for molecular splitting (sequences)
        # - splits: [1-test_fraction, test_fraction] for train/test ratio
        # - max_sim: maximum similarity threshold
        try:
            results = datasail(
                techniques=["M"],  # Molecular (sequence) splitting
                splits=[1 - test_fraction, test_fraction],
                names=["train", "test"],
                max_sim=identity,
                epsilon=0.05,  # Tolerance for achieving target split
                delta=0.05,  # Probability of failure
                stratified=True,  # Maintain class balance if labels present
                solver="GUROBI",  # Optimization solver (falls back to GLPK)
                output=str(output_dir),
                verbosity=0,
                e_type=["M"],  # Entity type: molecules
                e_data=[molecules],
                random_state=random_state,
            )

            # Parse results - DataSAIL returns dictionary with split assignments
            train_seqs = []
            test_seqs = []

            for seq_id, split_assignment in results["M"].items():
                if split_assignment == 0:  # train
                    train_seqs.append(seq_id)
                elif split_assignment == 1:  # test
                    test_seqs.append(seq_id)

            print(f"DataSAIL split: {len(train_seqs)} train, {len(test_seqs)} test "
                  f"({len(test_seqs)/(len(train_seqs)+len(test_seqs))*100:.1f}% test)")

            return train_seqs, test_seqs

        except Exception as e:
            raise RuntimeError(f"DataSAIL execution failed: {str(e)}")


def split_with_datasail_cli(
    fasta_path: Path,
    identity: float = 0.8,
    test_fraction: float = 0.2,
    random_state: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Alternative implementation using DataSAIL CLI.

    Use this if the Python API has compatibility issues.

    Args:
        fasta_path: Path to input FASTA file
        identity: Maximum sequence identity threshold
        test_fraction: Fraction of sequences for test set
        random_state: Random seed

    Returns:
        Tuple of (train_seq_ids, test_seq_ids)

    Raises:
        FileNotFoundError: If datasail CLI not found
        RuntimeError: If datasail execution fails
    """
    fasta_path = Path(fasta_path)

    # Check if datasail CLI is available
    try:
        subprocess.run(
            ["datasail", "--help"],
            capture_output=True,
            check=True
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "datasail CLI not found. Install with: pip install datasail"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)

        # Build command
        cmd = [
            "datasail",
            "--techniques", "M",
            "--splits", str(1 - test_fraction), str(test_fraction),
            "--names", "train", "test",
            "--max-sim", str(identity),
            "--epsilon", "0.05",
            "--output", str(output_dir),
            "--e-type", "M",
            "--e-data", str(fasta_path),
            "--random-state", str(random_state),
        ]

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output files
            train_file = output_dir / "train_M.txt"
            test_file = output_dir / "test_M.txt"

            train_seqs = []
            test_seqs = []

            if train_file.exists():
                with open(train_file) as f:
                    train_seqs = [line.strip() for line in f if line.strip()]

            if test_file.exists():
                with open(test_file) as f:
                    test_seqs = [line.strip() for line in f if line.strip()]

            print(f"DataSAIL CLI split: {len(train_seqs)} train, {len(test_seqs)} test")

            return train_seqs, test_seqs

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"DataSAIL CLI failed: {e.stderr}")
