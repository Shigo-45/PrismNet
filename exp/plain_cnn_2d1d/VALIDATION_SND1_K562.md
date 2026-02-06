# Validation Results - SND1_K562

**Date**: 2026-02-06
**Model**: Plain 2D-1D CNN (`evaluation/baselines_full/models/SND1_K562_plain_cnn_2d1d.pth`)
**Input**: `/home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv`

## Format Verification

✓ Saliency output: 3 columns (index, probability, saliency_matrix)
✓ HAR output: 4 columns (index, probability, start_pos, end_pos)
✓ Line counts: 15,002 sequences processed
✓ HAR positions in valid range [0, 101]
✓ HAR windows are exactly 20nt

## Sanity Checks

✓ All saliency scores non-negative (squared gradients)
✓ All probabilities in [0, 1] range
✓ HAR windows are 20nt (end = start + 20)

## Comparison with PrismNet

✓ Identical output format (compatible with motif analysis pipeline)
✓ Probability correlation: **0.654** (moderate positive correlation)
✓ Both models identify binding sites (probabilities vary across sequences)

**Prediction statistics:**
- Plain CNN: mean=0.402, std=0.451, range=[0.000, 1.000]
- PrismNet: mean=0.671, std=0.376, range=[0.000, 1.000]

**Analysis:** The 0.654 correlation indicates both models capture similar RNA binding patterns. PrismNet predicts higher binding probabilities on average (mean 0.671 vs 0.402), while Plain CNN shows greater prediction variance. Different HAR regions suggest architectural differences affect attention localization.

## Performance

- Saliency computation: ~58 minutes (15,002 sequences, batch size 64)
- HAR extraction: ~49 minutes (15,002 sequences)
- CPU-only execution (no GPU for gradient computation)

## Output Files

- Saliency: `out/saliency/SND1_K562_PlainCNN2D1D_pu.sal` (44MB, 15,002 lines)
- HAR: `out/har/SND1_K562_PlainCNN2D1D_pu.har` (299KB, 15,002 lines)
- Logs: `out/log_saliency_SND1_K562.txt`, `out/log_har_SND1_K562.txt`

## Bug Fixes Applied

During validation, fixed PyTorch compatibility issue in `prismnet/model/smoothgrad.py`:
- Updated `GuidedBackpropReLU` to use new-style static forward/backward methods
- Fixed deprecated `is` comparison for string matching

## Status

✅ Pipeline validated - ready for extension to other proteins
✅ Comparison with PrismNet complete (correlation: 0.654)

## Next Steps

1. Test on 2-3 additional proteins (TIA1_Hela, PTBP1_Hela)
2. Compare HAR position distributions between plain CNN and PrismNet
3. Locate EIF1 transcript data for case study reproduction
4. Scale to all 172 proteins for comprehensive ablation study
