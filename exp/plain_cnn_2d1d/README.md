# Plain CNN Saliency and HAR Extraction

This directory contains scripts for extracting saliency maps and High Attention Regions (HARs) from plain 2D-1D CNN models.

## Usage

### Saliency Maps

Extract numerical saliency scores for all sequences:

```bash
./saliency.sh <PROTEIN_NAME> <INFER_FILE.tsv>

# Example
./saliency.sh SND1_K562 /home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv
```

**Output**: `out/saliency/<PROTEIN>_plain_cnn_2d1d_<PROTEIN>.sal`

Format: `{index}\t{probability}\t{saliency_matrix_string}`

### High Attention Regions (HARs)

Extract 20nt windows with highest attention:

```bash
./har.sh <PROTEIN_NAME> <INFER_FILE.tsv>

# Example
./har.sh SND1_K562 /home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv
```

**Output**: `out/har/<PROTEIN>_plain_cnn_2d1d_<PROTEIN>.har`

Format: `{index}\t{probability}\t{start_pos}\t{end_pos}`

## Requirements

- Trained plain CNN model must exist: `evaluation/baselines_full/models/<PROTEIN>_plain_cnn_2d1d.pth`
- Inference file in TSV format (same format as training data)
- PyTorch environment with PrismNet dependencies

## Method

Uses `GuidedBackpropSmoothGrad` from `prismnet/model/smoothgrad.py` with identical hyperparameters to PrismNet:
- `x_stddev`: 0.015
- `t_stddev`: 0.015
- `nsamples`: 20
- `magnitude`: 2 (squared gradients)

## Comparison with PrismNet

To compare with PrismNet outputs:

```bash
# Generate PrismNet reference
cd /home/shigo-45/projects/PrismNet/exp/train_all
./saliency.sh SND1_K562 /path/to/SND1_K562.tsv
./har.sh SND1_K562 /path/to/SND1_K562.tsv

# Generate plain CNN outputs
cd /home/shigo-45/projects/PrismNet-eval-ablation/exp/plain_cnn_2d1d
./saliency.sh SND1_K562 /path/to/SND1_K562.tsv
./har.sh SND1_K562 /path/to/SND1_K562.tsv
```

Output formats are identical for direct comparison.
