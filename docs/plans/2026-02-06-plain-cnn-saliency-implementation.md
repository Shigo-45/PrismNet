# Plain CNN Saliency and HAR Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable saliency map and HAR extraction for plain 2D-1D CNN models using identical methodology to PrismNet

**Architecture:** Extend `tools/main.py` to support custom model paths and plain CNN architecture, create shell scripts in `exp/plain_cnn_2d1d/` that mirror PrismNet workflow, validate on SND1_K562

**Tech Stack:** PyTorch, GuidedBackpropSmoothGrad, existing PrismNet infrastructure

---

## Task 1: Modify tools/main.py to Support Custom Model Paths

**Files:**
- Modify: `tools/main.py:90-120` (argument parser section)
- Modify: `tools/main.py:160-165` (model loading section)

**Step 1: Add --model_path argument**

Add after line 95 (`parser.add_argument("--infer_file"...)`):

```python
parser.add_argument("--model_path",     type=str, help="custom model checkpoint path (overrides default)", default="")
```

**Step 2: Modify model loading logic**

Replace lines 162-164:

```python
if args.load_best:
    filename = model_path.format("best")
    print("Loading model: {}".format(filename))
```

With:

```python
if args.load_best:
    if args.model_path:
        filename = args.model_path
        print("Loading custom model: {}".format(filename))
    else:
        filename = model_path.format("best")
        print("Loading model: {}".format(filename))
```

**Step 3: Verify changes don't break existing behavior**

Run: `python tools/main.py --help | grep -A1 model_path`

Expected: Shows the new `--model_path` argument in help output

**Step 4: Commit**

```bash
git add tools/main.py
git commit -m "feat: add --model_path argument for custom model loading

Allows specifying custom checkpoint paths instead of default
{out_dir}/out/models/{identity}_best.pth location. Required for
loading plain CNN models from evaluation/baselines_full/models/."
```

---

## Task 2: Add PlainCNN2D1D Architecture Support

**Files:**
- Modify: `tools/main.py:13` (imports section)

**Step 1: Import PlainCNN2D1D**

Add after line 13 (`import prismnet.model as arch`):

```python
from prismnet_eval.ablation.baselines import PlainCNN2D1D
```

**Step 2: Register architecture in arch module**

Add after the import:

```python
# Register plain CNN for --arch PlainCNN2D1D
arch.PlainCNN2D1D = PlainCNN2D1D
```

**Step 3: Test import**

Run: `python -c "from prismnet_eval.ablation.baselines import PlainCNN2D1D; print('Import successful')"`

Expected: "Import successful"

**Step 4: Test architecture instantiation**

Run: `python -c "import prismnet.model as arch; from prismnet_eval.ablation.baselines import PlainCNN2D1D; arch.PlainCNN2D1D = PlainCNN2D1D; model = getattr(arch, 'PlainCNN2D1D')(mode='pu'); print('Model created:', type(model))"`

Expected: "Model created: <class 'prismnet_eval.ablation.baselines.PlainCNN2D1D'>"

**Step 5: Commit**

```bash
git add tools/main.py
git commit -m "feat: register PlainCNN2D1D architecture for --arch flag

Imports PlainCNN2D1D from ablation baselines and registers it
in prismnet.model namespace for dynamic instantiation via --arch."
```

---

## Task 3: Create exp/plain_cnn_2d1d Directory Structure

**Files:**
- Create: `exp/plain_cnn_2d1d/` (directory)
- Create: `exp/plain_cnn_2d1d/out/saliency/` (directory)
- Create: `exp/plain_cnn_2d1d/out/har/` (directory)

**Step 1: Create directory structure**

Run: `mkdir -p exp/plain_cnn_2d1d/out/saliency exp/plain_cnn_2d1d/out/har`

Expected: Directories created without errors

**Step 2: Verify structure**

Run: `tree exp/plain_cnn_2d1d`

Expected:
```
exp/plain_cnn_2d1d
└── out
    ├── har
    └── saliency
```

**Step 3: Create .gitkeep files for empty directories**

Run: `touch exp/plain_cnn_2d1d/out/saliency/.gitkeep exp/plain_cnn_2d1d/out/har/.gitkeep`

Expected: Files created

**Step 4: Commit**

```bash
git add exp/plain_cnn_2d1d/
git commit -m "feat: create exp/plain_cnn_2d1d directory structure

Mirrors exp/prismnet/ layout for plain CNN saliency and HAR outputs."
```

---

## Task 4: Write saliency.sh Script

**Files:**
- Create: `exp/plain_cnn_2d1d/saliency.sh`

**Step 1: Write saliency.sh**

```bash
#!/bin/bash
work_path=$(dirname $0)
name=$(basename $work_path)

p=$1                    # Protein name (e.g., SND1_K562)
infer_file=$2           # TSV file path

if [ -z "$p" ] || [ -z "$infer_file" ]; then
    echo "Usage: $0 <protein_name> <infer_file.tsv>"
    echo "Example: $0 SND1_K562 /path/to/SND1_K562.tsv"
    exit 1
fi

exp=$name               # "plain_cnn_2d1d"
model_file="evaluation/baselines_full/models/${p}_plain_cnn_2d1d.pth"

# Check if model exists
if [ ! -f "$model_file" ]; then
    echo "Error: Model not found: $model_file"
    exit 1
fi

# Check if inference file exists
if [ ! -f "$infer_file" ]; then
    echo "Error: Inference file not found: $infer_file"
    exit 1
fi

echo "Running saliency extraction for $p using plain CNN..."
echo "Model: $model_file"
echo "Input: $infer_file"

python -u tools/main.py \
    --arch PlainCNN2D1D \
    --load_best \
    --model_path $model_file \
    --saliency \
    --infer_file $infer_file \
    --p_name $p \
    --out_dir $work_path \
    --exp_name $exp \
    ${@:3} | tee $work_path/out/log_saliency_${p}.txt
```

**Step 2: Make script executable**

Run: `chmod +x exp/plain_cnn_2d1d/saliency.sh`

Expected: Script is executable

**Step 3: Test script help message**

Run: `exp/plain_cnn_2d1d/saliency.sh`

Expected: Usage message displayed

**Step 4: Commit**

```bash
git add exp/plain_cnn_2d1d/saliency.sh
git commit -m "feat: add saliency.sh for plain CNN models

Shell script to extract saliency maps from plain 2D-1D CNN models.
Includes error checking for model and input file existence."
```

---

## Task 5: Write har.sh Script

**Files:**
- Create: `exp/plain_cnn_2d1d/har.sh`

**Step 1: Write har.sh**

```bash
#!/bin/bash
work_path=$(dirname $0)
name=$(basename $work_path)

p=$1                    # Protein name (e.g., SND1_K562)
infer_file=$2           # TSV file path

if [ -z "$p" ] || [ -z "$infer_file" ]; then
    echo "Usage: $0 <protein_name> <infer_file.tsv>"
    echo "Example: $0 SND1_K562 /path/to/SND1_K562.tsv"
    exit 1
fi

exp=$name               # "plain_cnn_2d1d"
model_file="evaluation/baselines_full/models/${p}_plain_cnn_2d1d.pth"

# Check if model exists
if [ ! -f "$model_file" ]; then
    echo "Error: Model not found: $model_file"
    exit 1
fi

# Check if inference file exists
if [ ! -f "$infer_file" ]; then
    echo "Error: Inference file not found: $infer_file"
    exit 1
fi

echo "Running HAR extraction for $p using plain CNN..."
echo "Model: $model_file"
echo "Input: $infer_file"

python -u tools/main.py \
    --arch PlainCNN2D1D \
    --load_best \
    --model_path $model_file \
    --har \
    --infer_file $infer_file \
    --p_name $p \
    --out_dir $work_path \
    --exp_name $exp \
    ${@:3} | tee $work_path/out/log_har_${p}.txt
```

**Step 2: Make script executable**

Run: `chmod +x exp/plain_cnn_2d1d/har.sh`

Expected: Script is executable

**Step 3: Test script help message**

Run: `exp/plain_cnn_2d1d/har.sh`

Expected: Usage message displayed

**Step 4: Commit**

```bash
git add exp/plain_cnn_2d1d/har.sh
git commit -m "feat: add har.sh for plain CNN models

Shell script to extract High Attention Regions (20nt windows) from
plain 2D-1D CNN models. Includes error checking for model and input."
```

---

## Task 6: Create README Documentation

**Files:**
- Create: `exp/plain_cnn_2d1d/README.md`

**Step 1: Write README**

```markdown
# Plain CNN Saliency and HAR Extraction

This directory contains scripts for extracting saliency maps and High Attention Regions (HARs) from plain 2D-1D CNN models.

## Usage

### Saliency Maps

Extract numerical saliency scores for all sequences:

\`\`\`bash
./saliency.sh <PROTEIN_NAME> <INFER_FILE.tsv>

# Example
./saliency.sh SND1_K562 /home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv
\`\`\`

**Output**: `out/saliency/<PROTEIN>_plain_cnn_2d1d_<PROTEIN>.sal`

Format: `{index}\t{probability}\t{saliency_matrix_string}`

### High Attention Regions (HARs)

Extract 20nt windows with highest attention:

\`\`\`bash
./har.sh <PROTEIN_NAME> <INFER_FILE.tsv>

# Example
./har.sh SND1_K562 /home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv
\`\`\`

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

\`\`\`bash
# Generate PrismNet reference
cd /home/shigo-45/projects/PrismNet/exp/train_all
./saliency.sh SND1_K562 /path/to/SND1_K562.tsv
./har.sh SND1_K562 /path/to/SND1_K562.tsv

# Generate plain CNN outputs
cd /home/shigo-45/projects/PrismNet-eval-ablation/exp/plain_cnn_2d1d
./saliency.sh SND1_K562 /path/to/SND1_K562.tsv
./har.sh SND1_K562 /path/to/SND1_K562.tsv
\`\`\`

Output formats are identical for direct comparison.
```

**Step 2: Commit**

```bash
git add exp/plain_cnn_2d1d/README.md
git commit -m "docs: add README for plain CNN saliency extraction

Documents usage, output formats, and comparison with PrismNet."
```

---

## Task 7: Validate Installation - Test on SND1_K562

**Files:**
- Input: `evaluation/baselines_full/models/SND1_K562_plain_cnn_2d1d.pth` (should exist)
- Input: `/home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv` (should exist)
- Output: `exp/plain_cnn_2d1d/out/saliency/SND1_K562_plain_cnn_2d1d_SND1_K562.sal`
- Output: `exp/plain_cnn_2d1d/out/har/SND1_K562_plain_cnn_2d1d_SND1_K562.har`

**Step 1: Verify plain CNN model exists**

Run: `ls -lh evaluation/baselines_full/models/SND1_K562_plain_cnn_2d1d.pth`

Expected: File exists with reasonable size (several MB)

**Step 2: Verify inference data exists**

Run: `ls -lh /home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv`

Expected: File exists (should be ~10MB based on git status earlier)

**Step 3: Run saliency extraction**

Run: `cd /home/shigo-45/projects/PrismNet-eval-ablation && exp/plain_cnn_2d1d/saliency.sh SND1_K562 /home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv`

Expected:
- Script runs without errors
- Progress output showing batch processing
- Output file created: `exp/plain_cnn_2d1d/out/saliency/SND1_K562_plain_cnn_2d1d_SND1_K562.sal`

**Step 4: Verify saliency output format**

Run: `head -3 exp/plain_cnn_2d1d/out/saliency/SND1_K562_plain_cnn_2d1d_SND1_K562.sal`

Expected: Three lines with format `{index}\t{prob}\t{saliency_matrix}`
- Index is integer
- Probability is float in [0, 1]
- Saliency matrix is space-separated values

**Step 5: Count output lines**

Run: `wc -l exp/plain_cnn_2d1d/out/saliency/SND1_K562_plain_cnn_2d1d_SND1_K562.sal`

Expected: Number of lines matches input TSV (minus header if present)

**Step 6: Run HAR extraction**

Run: `exp/plain_cnn_2d1d/har.sh SND1_K562 /home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv`

Expected:
- Script runs without errors
- Output file created: `exp/plain_cnn_2d1d/out/har/SND1_K562_plain_cnn_2d1d_SND1_K562.har`

**Step 7: Verify HAR output format**

Run: `head -10 exp/plain_cnn_2d1d/out/har/SND1_K562_plain_cnn_2d1d_SND1_K562.har`

Expected: Lines with format `{index}\t{prob}\t{start}\t{end}`
- Start and end positions are integers in range [0, 101]
- End = start + 20

**Step 8: Basic sanity check - saliency scores non-negative**

Run: `awk '{print $3}' exp/plain_cnn_2d1d/out/saliency/SND1_K562_plain_cnn_2d1d_SND1_K562.sal | head -1 | tr ' ' '\n' | awk 'NF && $1 < 0 {print "NEGATIVE FOUND"; exit 1}'`

Expected: No output (no negative values found)

**Step 9: No commit needed** (outputs are gitignored)

---

## Task 8: Generate PrismNet Reference Data

**Files:**
- Output: `/home/shigo-45/projects/PrismNet/exp/train_all/out/saliency/SND1_K562_train_all_SND1_K562.sal`
- Output: `/home/shigo-45/projects/PrismNet/exp/train_all/out/har/SND1_K562_train_all_SND1_K562.har`

**Step 1: Run PrismNet saliency**

Run: `cd /home/shigo-45/projects/PrismNet && exp/train_all/saliency.sh SND1_K562 /home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv`

Expected: Saliency file generated

**Step 2: Run PrismNet HAR**

Run: `cd /home/shigo-45/projects/PrismNet && exp/train_all/har.sh SND1_K562 /home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv`

Expected: HAR file generated

**Step 3: Verify PrismNet outputs exist**

Run: `ls -lh /home/shigo-45/projects/PrismNet/exp/train_all/out/saliency/*SND1_K562*.sal /home/shigo-45/projects/PrismNet/exp/train_all/out/har/*SND1_K562*.har`

Expected: Both files exist with reasonable sizes

**Step 4: No commit needed** (different repo)

---

## Task 9: Compare Output Formats

**Files:**
- Compare: Plain CNN vs PrismNet saliency files
- Compare: Plain CNN vs PrismNet HAR files

**Step 1: Compare saliency line counts**

Run:
```bash
echo "Plain CNN:" && wc -l /home/shigo-45/projects/PrismNet-eval-ablation/exp/plain_cnn_2d1d/out/saliency/*SND1_K562*.sal
echo "PrismNet:" && wc -l /home/shigo-45/projects/PrismNet/exp/train_all/out/saliency/*SND1_K562*.sal
```

Expected: Same line counts

**Step 2: Compare HAR line counts**

Run:
```bash
echo "Plain CNN:" && wc -l /home/shigo-45/projects/PrismNet-eval-ablation/exp/plain_cnn_2d1d/out/har/*SND1_K562*.har
echo "PrismNet:" && wc -l /home/shigo-45/projects/PrismNet/exp/train_all/out/har/*SND1_K562*.har
```

Expected: Same line counts

**Step 3: Compare saliency format (column structure)**

Run:
```bash
head -1 /home/shigo-45/projects/PrismNet-eval-ablation/exp/plain_cnn_2d1d/out/saliency/*SND1_K562*.sal | awk -F'\t' '{print "Columns:", NF}'
head -1 /home/shigo-45/projects/PrismNet/exp/train_all/out/saliency/*SND1_K562*.sal | awk -F'\t' '{print "Columns:", NF}'
```

Expected: Both show "Columns: 3"

**Step 4: Compare HAR format (column structure)**

Run:
```bash
head -1 /home/shigo-45/projects/PrismNet-eval-ablation/exp/plain_cnn_2d1d/out/har/*SND1_K562*.har | awk -F'\t' '{print "Columns:", NF}'
head -1 /home/shigo-45/projects/PrismNet/exp/train_all/out/har/*SND1_K562*.har | awk -F'\t' '{print "Columns:", NF}'
```

Expected: Both show "Columns: 4"

**Step 5: Check probability correlation**

Run:
```bash
python -c "
import numpy as np
plain = np.loadtxt('/home/shigo-45/projects/PrismNet-eval-ablation/exp/plain_cnn_2d1d/out/saliency/SND1_K562_plain_cnn_2d1d_SND1_K562.sal', usecols=1)
prism = np.loadtxt('/home/shigo-45/projects/PrismNet/exp/train_all/out/saliency/SND1_K562_train_all_SND1_K562.sal', usecols=1)
corr = np.corrcoef(plain, prism)[0,1]
print(f'Probability correlation: {corr:.3f}')
assert corr > 0.5, f'Low correlation: {corr}'
"
```

Expected: Correlation > 0.5 (both models trained on same data)

**Step 6: No commit needed** (validation only)

---

## Task 10: Create Validation Summary

**Files:**
- Create: `exp/plain_cnn_2d1d/VALIDATION_SND1_K562.md`

**Step 1: Write validation summary**

```markdown
# Validation Results - SND1_K562

**Date**: 2026-02-06
**Model**: Plain 2D-1D CNN (`evaluation/baselines_full/models/SND1_K562_plain_cnn_2d1d.pth`)
**Input**: `/home/shigo-45/projects/PrismNet/data/clip_data/SND1_K562.tsv`

## Format Verification

✓ Saliency output: 3 columns (index, probability, saliency_matrix)
✓ HAR output: 4 columns (index, probability, start_pos, end_pos)
✓ Line counts match between Plain CNN and PrismNet
✓ HAR positions in valid range [0, 101]

## Sanity Checks

✓ All saliency scores non-negative (squared gradients)
✓ All probabilities in [0, 1] range
✓ HAR windows are 20nt (end = start + 20)

## Comparison with PrismNet

✓ Identical output format (compatible with motif analysis pipeline)
✓ Probability correlation: [INSERT VALUE FROM STEP 9.5]
✓ Both models identify binding sites (probabilities vary across sequences)

## Output Files

- Saliency: `out/saliency/SND1_K562_plain_cnn_2d1d_SND1_K562.sal`
- HAR: `out/har/SND1_K562_plain_cnn_2d1d_SND1_K562.har`
- Logs: `out/log_saliency_SND1_K562.txt`, `out/log_har_SND1_K562.txt`

## Status

✅ Pipeline validated - ready for extension to other proteins

## Next Steps

1. Test on 2-3 additional proteins (TIA1_Hela, PTBP1_Hela)
2. Compare HAR position distributions between plain CNN and PrismNet
3. Locate EIF1 transcript data for case study reproduction
```

**Step 2: Update with actual correlation value**

(Insert the correlation value from Task 9 Step 5)

**Step 3: Commit**

```bash
git add exp/plain_cnn_2d1d/VALIDATION_SND1_K562.md
git commit -m "docs: add validation results for SND1_K562

Documents successful saliency and HAR extraction from plain CNN,
confirms format compatibility with PrismNet."
```

---

## Success Criteria

✅ `tools/main.py` supports `--model_path` and `--arch PlainCNN2D1D`
✅ `exp/plain_cnn_2d1d/` directory structure created
✅ `saliency.sh` and `har.sh` scripts functional
✅ SND1_K562 saliency and HAR files generated successfully
✅ Output format matches PrismNet (3 and 4 columns respectively)
✅ Sanity checks pass (non-negative scores, valid ranges)
✅ Probability correlation with PrismNet > 0.5
✅ Documentation complete (README and validation summary)

## Next Steps (Out of Scope)

- Test on additional proteins (TIA1_Hela, PTBP1_Hela, etc.)
- Create batch processing script for all 172 proteins
- Implement saliency image generation (`saliencyimg.sh`)
- Statistical comparison of HAR distributions
- Reproduce EIF1 case study from paper
