# Evaluation Output Directory

This directory contains results from the homology-aware data splitting evaluation.
Binary and generated files (`.h5`, `.fasta`, `.log`, `.probs`) are excluded from
version control. Only reports, summaries, and metrics files are committed.

## Directory Structure

```
evaluation/
├── all_datasets/              # Full-dataset leakage analysis (172 proteins)
│   ├── splitting_evaluation_summary.json   # Per-dataset homology stats
│   ├── ALL_DATASETS_REPORT.md             # Human-readable summary
│   ├── datasets_clean.txt                 # Proteins with no leakage
│   └── datasets_with_leakage.txt          # Proteins with homology leakage
├── cdhit_validation/          # TIA1_Hela CD-HIT 80% split vs original
│   ├── TIA1_Hela_original/    # Model trained on original random split
│   └── TIA1_Hela_cdhit80/    # Model trained on CD-HIT homology-aware split
├── full_eval/                 # Per-protein CD-HIT splits
└── test/                      # Test splits used during development
```

## Regenerating Outputs

### Step 1: Leakage analysis (all 172 proteins)

```bash
python tools/eval_splitting.py analyze \
    --data-dir data/clip_data \
    --output evaluation/all_datasets/splitting_evaluation_summary.json
```

### Step 2: CD-HIT homology-aware splits

Requires `cd-hit` to be installed and on `PATH`.

```bash
# Single protein example
python tools/eval_splitting.py split \
    --input data/clip_data/TIA1_Hela.h5 \
    --output evaluation/full_eval/TIA1_Hela_cdhit80.h5 \
    --identity 0.80

# All proteins
for h5 in data/clip_data/*.h5; do
    name=$(basename "$h5" .h5)
    python tools/eval_splitting.py split \
        --input "$h5" \
        --output "evaluation/full_eval/${name}_cdhit80.h5" \
        --identity 0.80
done
```

### Step 3: CD-HIT validation (TIA1_Hela model comparison)

Train and evaluate models on original vs CD-HIT split:

```bash
tools/train_cdhit_comparison.sh
```

### Step 4: Regenerate figures

```bash
python visualization/create_figures.py
```

Figures are saved to `visualization/figures/` (also excluded from git).
