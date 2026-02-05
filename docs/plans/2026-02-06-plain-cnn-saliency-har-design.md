# Plain CNN Saliency Map and HAR Extraction Design

**Date**: 2026-02-06
**Purpose**: Extract saliency maps and High Attention Regions (HARs) from plain 2D-1D CNN models to prove comparability with full PrismNet

## Objective

Reproduce the PrismNet paper's saliency map extraction and HAR identification methodology for plain CNN models. This enables direct comparison between plain 2D-1D CNN architecture and full PrismNet with SE blocks and residual connections, using identical analysis methods.

## Background

- PrismNet paper demonstrated binding site interpretation through saliency maps and 20nt High Attention Regions (HARs)
- Case study: SND1 binding on EIF1 transcripts in K562 cells
- Plain CNN models trained as ablation baselines exist in `evaluation/baselines_full/models/`
- Need to prove plain architecture is comparable through interpretability analysis

## Architecture Overview

Create new experiment directory mirroring PrismNet structure:

```
exp/plain_cnn_2d1d/
├── saliency.sh          # Compute saliency maps
├── har.sh               # Compute high attention regions
└── out/
    ├── saliency/        # .sal files (numerical scores)
    └── har/             # .har files (20nt window positions)
```

### Key Design Principles

1. **Reuse proven implementations**: Use existing `compute_saliency()` and `compute_high_attention_region()` from `prismnet/engine/train_loop.py` without modification
2. **Identical methodology**: Ensure plain CNN saliency is computed exactly like PrismNet's (GuidedBackpropSmoothGrad, same hyperparameters)
3. **Compatible output format**: Generate files in same format for downstream motif analysis tools
4. **Validation first**: Start with SND1_K562 test set, extend to EIF1 case study data later

## Implementation Details

### Shell Script Structure

Both `saliency.sh` and `har.sh` follow this pattern:

```bash
#!/bin/bash
work_path=$(dirname $0)
name=$(basename $work_path)

p=$1                    # Protein name (e.g., SND1_K562)
infer_file=$2           # TSV file path

exp=$name               # "plain_cnn_2d1d"

python -u tools/main.py \
    --arch PlainCNN2D1D \
    --load_best \
    --model_path evaluation/baselines_full/models/${p}_plain_cnn_2d1d.pth \
    --saliency \          # (or --har for har.sh)
    --infer_file $infer_file \
    --p_name $p \
    --out_dir $work_path \
    --exp_name $exp \
    ${@:3} | tee $work_path/out/log.txt
```

**Key differences from PrismNet scripts**:
- `--arch PlainCNN2D1D`: Specify plain CNN architecture
- `--model_path`: Explicitly point to plain CNN model checkpoint (not default PrismNet location)
- Identity naming: Outputs named like `SND1_K562_plain_cnn_2d1d` instead of `SND1_K562_PrismNet_pu`

### Required Code Modifications

**In `tools/main.py`**:

1. Add `--model_path` argument to override default checkpoint location
2. Add architecture registry to instantiate `PlainCNN2D1D` when `--arch` is specified
3. Import plain CNN from ablation framework:
   ```python
   from prismnet_eval.ablation.baselines import PlainCNN2D1D
   ```

**No changes needed**:
- `prismnet/engine/train_loop.py`: Functions are architecture-agnostic
- `prismnet/model/smoothgrad.py`: Works with any PyTorch model

## Data Flow

### Input Processing

1. **Model Loading**: Load trained plain CNN from `evaluation/baselines_full/models/SND1_K562_plain_cnn_2d1d.pth`
2. **Data Loading**: Use `SeqicSHAPE` loader in inference mode with `SND1_K562.tsv` (full dataset)
3. **Batch Processing**: Process in batches (default 64)

### Saliency Computation

The `compute_saliency()` function:
1. Instantiates `GuidedBackpropSmoothGrad` with plain CNN model
2. For each batch:
   - Forward pass → predictions
   - `sgrad.get_batch_gradients()` → compute gradients
   - Extract saliency for all 5 features (ACGU + icSHAPE)
3. Save to `.sal` file with format:
   ```
   {index}\t{probability}\t{saliency_matrix_as_string}
   ```

**Output**: `exp/plain_cnn_2d1d/out/saliency/SND1_K562_plain_cnn_2d1d_SND1_K562.sal`

### HAR Computation

The `compute_high_attention_region()` function:
1. Compute saliency maps (as above)
2. Sum across all 5 feature dimensions → position-wise attention scores
3. Sliding 20nt window → find highest scoring region
4. Save to `.har` file with format:
   ```
   {index}\t{probability}\t{start_pos}\t{end_pos}
   ```

**Output**: `exp/plain_cnn_2d1d/out/har/SND1_K562_plain_cnn_2d1d_SND1_K562.har`

### Compatibility

Output format is identical to PrismNet, enabling:
- Direct comparison of saliency distributions
- Use of existing motif analysis pipeline (`motif_construct/saliency_motif.pl`)
- HAR position overlap analysis

## Testing and Validation Strategy

### Phase 1: Initial Validation (SND1_K562)

1. **Generate PrismNet reference data**:
   ```bash
   cd /home/shigo-45/projects/PrismNet/exp/train_all
   ./saliency.sh SND1_K562 /path/to/SND1_K562.tsv
   ./har.sh SND1_K562 /path/to/SND1_K562.tsv
   ```

2. **Generate plain CNN data**:
   ```bash
   cd /home/shigo-45/projects/PrismNet-eval-ablation/exp/plain_cnn_2d1d
   ./saliency.sh SND1_K562 /path/to/SND1_K562.tsv
   ./har.sh SND1_K562 /path/to/SND1_K562.tsv
   ```

3. **Validation checks**:
   - **Format verification**: Identical column structure, row counts match input
   - **Sanity checks**:
     - Saliency scores are non-negative (squared gradients)
     - HAR positions within valid range [0, 101]
     - Prediction probabilities in [0, 1]
   - **Comparison**:
     - Saliency score distributions (plain CNN vs PrismNet)
     - HAR position distributions
     - Prediction probability correlation

### Success Criteria

- Plain CNN outputs have identical format to PrismNet
- Both models identify meaningful binding regions (non-uniform HAR distributions)
- Prediction probabilities show reasonable correlation (both trained on same data)

### Phase 2: Extension

Once SND1_K562 validated:
1. Test on 2-3 additional proteins (e.g., TIA1_Hela, PTBP1_Hela)
2. Verify workflow generalizes across datasets
3. Search for and analyze EIF1 transcript data for case study reproduction

## Error Handling

The scripts should handle these failure modes:

1. **Missing model file**: Check existence before loading
   - Error: `Model not found: {path}`

2. **Missing inference file**: Verify TSV exists and is readable
   - Error: `Inference file not found: {path}`

3. **Model architecture mismatch**: PyTorch will raise error if checkpoint incompatible
   - Let PyTorch error propagate with clear message

4. **Empty dataset**: If TSV has no valid sequences
   - Print warning, create empty output files (matches PrismNet behavior)

## Implementation Order

1. **Modify `tools/main.py`**: Add `--model_path` argument and `PlainCNN2D1D` architecture support
2. **Create directory structure**: `exp/plain_cnn_2d1d/` with `out/saliency/` and `out/har/`
3. **Write shell scripts**: `saliency.sh` and `har.sh`
4. **Test on SND1_K562**: Generate both plain CNN and PrismNet outputs
5. **Validate outputs**: Run all validation checks
6. **Document usage**: Add README to `exp/plain_cnn_2d1d/`

## Out of Scope (Future Extensions)

- Batch processing script for all 172 proteins
- Saliency image generation (`saliencyimg.sh` with PDF visualizations)
- Direct comparison tools (automated diff analysis)
- Statistical significance testing of HAR overlap
- Full EIF1 case study reproduction (requires locating/formatting EIF1 data)

## Dependencies

- Existing: PyTorch, GuidedBackpropSmoothGrad, SeqicSHAPE loader
- No new dependencies required
- Reuses all existing saliency computation infrastructure

## Success Metrics

1. **Technical**: Scripts execute without errors, outputs match expected format
2. **Scientific**: Plain CNN identifies biologically meaningful binding regions comparable to PrismNet
3. **Reproducibility**: Can regenerate paper's SND1-EIF1 case study with plain CNN architecture
