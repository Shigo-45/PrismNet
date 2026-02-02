# CD-HIT Split Validation - Training in Progress

**Status:** Training models on original and CD-HIT splits for comparison
**Started:** 2026-02-02 22:10
**Expected completion:** ~2-3 hours for all 6 models (3 datasets × 2 splits)

---

## Training Plan

### Datasets to Validate
1. **TIA1_Hela** - RNA-binding protein
2. **IGF2BP1_K562** - Insulin-like growth factor 2 mRNA-binding protein
3. **SRSF1_HepG2** - Serine/arginine-rich splicing factor 1

### For Each Dataset
- Train on **original split** (random)
- Train on **CD-HIT split** (homology-aware, 0.8 threshold)
- Compare AUC, accuracy, and loss

---

## Current Progress

### TIA1_Hela - Original Split (IN PROGRESS)

**Training started:** 22:10
**Current status:** Epoch 11/200

**Early results:**
```
Epoch 1:  Train AUC: 0.9244, Test AUC: 0.9453
Epoch 5:  Train AUC: 0.9537, Test AUC: 0.9513
Epoch 9:  Train AUC: 0.9607, Test AUC: 0.9538
Epoch 11: Train AUC: 0.9617, Test AUC: 0.9535
```

**Observations:**
- Model is learning well
- Test AUC ~0.95 (excellent performance)
- No signs of overfitting (test AUC stable)

---

## Expected Results

### Hypothesis
Based on homology analysis:
- **91.9% of datasets have 0% leakage** → no change expected
- **Mean identity 25.5%** (random baseline) → no homology to exploit
- **CD-HIT splits nearly identical** → minimal performance difference

### Prediction
**Expected AUC difference: 0-2%**

| Dataset | Original AUC | CD-HIT AUC | Difference | Status |
|---------|--------------|------------|------------|--------|
| TIA1_Hela | ~0.95 | ~0.93-0.95 | 0-2% | Training original... |
| IGF2BP1_K562 | TBD | TBD | 0-2% | Pending |
| SRSF1_HepG2 | TBD | TBD | 0-2% | Pending |

---

## What This Validates

### If AUC difference is 0-2% (Expected)
✅ **Confirms:** Model learns genuine binding patterns, not memorization
✅ **Confirms:** Original splits are valid (no leakage inflation)
✅ **Confirms:** Homology-aware splitting is unnecessary for PrismNet

### If AUC difference is 2-5% (Unlikely)
⚠️ **Suggests:** Some overfitting in original splits
⚠️ **Action:** Consider CD-HIT splitting for critical applications
⚠️ **Still acceptable:** Within reasonable variance

### If AUC difference is >5% (Very Unlikely)
❌ **Would indicate:** Systematic overfitting to similar sequences
❌ **Would require:** Re-evaluation of all datasets
❌ **But:** Data characteristics make this extremely unlikely

---

## Training Configuration

**Model:** PrismNet (58,189 parameters)
**Architecture:** CNN with residual blocks and SE attention
**Optimizer:** Adam with warmup scheduler
**Learning rate:** 0.001 (with warmup)
**Batch size:** 64
**Epochs:** 200 (with early stopping patience=20)
**Loss:** BCEWithLogitsLoss (pos_weight=2)

**Data:**
- Train: 12,002 sequences (8,002 negative, 4,000 positive)
- Test: 3,000 sequences (2,000 negative, 1,000 positive)
- Sequence length: 101 bp
- Features: 5 (4 nucleotides + 1 structure)

---

## Timeline

| Time | Task | Status |
|------|------|--------|
| 22:10 | Start TIA1_Hela original | ✅ In progress |
| ~22:40 | Complete TIA1_Hela original | Pending |
| ~22:45 | Start TIA1_Hela CD-HIT | Pending |
| ~23:15 | Complete TIA1_Hela CD-HIT | Pending |
| ~23:20 | Start IGF2BP1_K562 original | Pending |
| ~23:50 | Complete IGF2BP1_K562 original | Pending |
| ~23:55 | Start IGF2BP1_K562 CD-HIT | Pending |
| ~00:25 | Complete IGF2BP1_K562 CD-HIT | Pending |
| ~00:30 | Start SRSF1_HepG2 original | Pending |
| ~01:00 | Complete SRSF1_HepG2 original | Pending |
| ~01:05 | Start SRSF1_HepG2 CD-HIT | Pending |
| ~01:35 | Complete SRSF1_HepG2 CD-HIT | Pending |
| ~01:40 | Generate comparison report | Pending |

**Total estimated time:** ~3.5 hours

---

## Monitoring

**Training logs:** `/tmp/claude-1000/-home-shigo-45-projects-PrismNet/tasks/b7fadb4.output`
**Output directory:** `exp/prismnet/out/`
**Results will be saved to:** `evaluation/cdhit_validation/`

**Check progress:**
```bash
tail -f /tmp/claude-1000/-home-shigo-45-projects-PrismNet/tasks/b7fadb4.output
```

---

## Next Steps

1. ✅ Training TIA1_Hela original (in progress)
2. ⏳ Train TIA1_Hela CD-HIT
3. ⏳ Train IGF2BP1_K562 (both splits)
4. ⏳ Train SRSF1_HepG2 (both splits)
5. ⏳ Compare results and generate report
6. ⏳ Commit findings

---

**Last updated:** 2026-02-02 22:15
**Status:** Training in progress, check back in ~30 minutes for first results
