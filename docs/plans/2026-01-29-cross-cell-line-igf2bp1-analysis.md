# Cross-Cell-Line IGF2BP1 Analysis

## Objective

Reproduce the PrismNet paper's proof-of-principle analysis:
- Train on IGF2BP1 K562 eCLIP + K562 icSHAPE
- Predict on HepG2 using HepG2 icSHAPE
- Evaluate predictions against HepG2 eCLIP ground truth

## Data Sources

| File | Description |
|------|-------------|
| `data/clip_data/IGF2BP1_K562.tsv` | Existing K562 training data |
| `data/ENCFF442USD.bed` | IGF2BP1 HepG2 IDR peaks (4,459) |
| `icSHAPE/HepG2-plus.bw` | HepG2 icSHAPE (+ strand) |
| `icSHAPE/HepG2-minus.bw` | HepG2 icSHAPE (- strand) |
| `data/reference/hg38.fa` | Reference genome |
| `data/gencode.v44.annotation.gtf` | Transcript annotations |

## Methodology (from paper)

### Positive Samples
- Take eCLIP peaks, unify length to 101nt (expand/trim from center)
- Keep top 5,000 peaks by signal strength
- Require ≥40% icSHAPE coverage (≥40 of 101 positions have values)

### Negative Samples
- Random 101nt regions from transcriptome
- Require ≥40% icSHAPE coverage
- Avoid overlap with binding regions
- Sample 10,000 negatives

## Implementation Steps

1. **Create preprocessing script** (`tools/create_hepg2_dataset.py`)
   - Read IDR peaks from BED
   - Extract 101nt sequences from genome
   - Query icSHAPE BigWig for structure values
   - Generate negative samples from transcriptome
   - Output TSV in PrismNet format

2. **Generate HepG2 dataset**
   - Run preprocessing to create `IGF2BP1_HepG2.tsv`
   - Convert to H5 format

3. **Train model on K562**
   - Use existing `IGF2BP1_K562.h5`
   - Run training script

4. **Predict on HepG2**
   - Run inference with K562-trained model
   - Use HepG2 icSHAPE structure data

5. **Evaluate**
   - Compare predictions to HepG2 eCLIP ground truth
   - Calculate AUC, precision, recall

## Output Files

- `data/clip_data/IGF2BP1_HepG2.tsv` - Processed HepG2 dataset
- `data/clip_data/IGF2BP1_HepG2.h5` - H5 format for inference
- `exp/prismnet/out/models/IGF2BP1_K562_best.pth` - Trained model
- `exp/prismnet/out/infer/IGF2BP1_HepG2.probs` - Predictions

## Date

2026-01-29
