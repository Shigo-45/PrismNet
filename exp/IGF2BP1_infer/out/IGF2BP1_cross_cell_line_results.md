# IGF2BP1 Cross-Cell-Line Prediction Results

## Experiment Overview

Reproduction of PrismNet paper proof-of-principle analysis:
- **Training**: IGF2BP1 eCLIP from K562 cells + K562 icSHAPE structure data
- **Testing**: IGF2BP1 eCLIP from HepG2 cells + HepG2 icSHAPE structure data

## Data Summary

### Training Data (K562)
- Source: Existing `IGF2BP1_K562.h5` from PrismNet repository
- Pre-processed with K562 icSHAPE data

### Testing Data (HepG2)
- eCLIP peaks: ENCFF442USD (IDR-filtered, 4,459 peaks)
- icSHAPE: HepG2-plus.bw, HepG2-minus.bw
- Final dataset: 1,677 positives + 10,000 negatives = 11,677 samples

Note: Only 1,677 of 4,459 peaks (37.6%) had sufficient icSHAPE coverage (>=40%)

## Model Configuration

- Architecture: PrismNet (58,189 parameters)
- Input mode: `pu` (protein + structure)
- Sequence length: 101 nt
- Trained with default hyperparameters

## Results

### Primary Metrics

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.7861** |
| **AUC-PR** | **0.4188** |

### Classification Metrics (threshold = 0.5)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7247 |
| Precision | 0.3009 |
| Recall | 0.6929 |
| F1-score | 0.4196 |

### Prediction Distribution

- Positive predictions (prob >= 0.5): 3,862 / 11,677
- Mean probability for true positives: 0.6758
- Mean probability for true negatives: 0.2990

## Interpretation

1. **AUC-ROC of 0.79** indicates good discriminative ability across cell types
2. The model successfully transfers knowledge from K562 to HepG2
3. Lower AUC-PR (0.42) reflects class imbalance (14.4% positives)
4. High recall (69%) shows model captures most true binding sites
5. Lower precision (30%) indicates some false positives, expected for cross-cell prediction

## Files Generated

- Model: `exp/prismnet/out/models/IGF2BP1_K562_PrismNet_pu_best.pth`
- Predictions: `exp/prismnet/out/infer/IGF2BP1_K562_PrismNet_pu_IGF2BP1_HepG2.tsv.probs`
- HepG2 dataset: `data/clip_data/IGF2BP1_HepG2.tsv`

## Date

2026-01-29
