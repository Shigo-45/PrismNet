# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Objective

Help users understand and reproduce the work of PrismNet: a PyTorch deep learning framework for predicting dynamic cellular protein-RNA interactions using in vivo RNA structure data (icSHAPE). Published in Cell Research (2021).

**Important**: When assisting users, always show commands and explain what they do rather than running them automatically. Let users review and execute commands themselves.

## Installation and Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

**Requirements**: Python 3.6+, PyTorch 2.0.0+, CUDA support recommended

## Data

Training data (172 protein samples) is already available in `data/clip_data/`. The data is in TSV format and needs to be converted to HDF5 format before training.

To prepare HDF5 datasets for binary classification:
```bash
tools/gdata_bin.sh
```

This processes all proteins listed in `data/clip_data/all.list` and creates corresponding `.h5` files in `data/clip_data/`.

## Training Commands

All training is performed via shell scripts in `exp/` directories. The standard pattern is:

```bash
# Train single protein model
exp/prismnet/train.sh <PROTEIN_NAME> <DATA_DIR>
# Example: exp/prismnet/train.sh TIA1_Hela clip_data

# Train all proteins
exp/prismnet/train_all.sh

# Monitor with TensorBoard (add -tfboard flag to train.sh)
tensorboard --logdir exp/prismnet/out/tfb
```

Training hyperparameters are configured in the shell scripts, which call `tools/main.py` with arguments.

**IMPORTANT - Resource Management**:
- **Do NOT run more than 3 PrismNet model training jobs in parallel** (including background jobs)
- Check GPU status with `nvidia-smi` before starting training
- Check RAM with `free -h` before starting training
- Each training job can consume significant GPU memory and system resources

**Key training options** (edit in train.sh):
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 64)
- `--nepochs`: Number of epochs (default: 200)
- `--early_stopping`: Early stopping patience (default: 20)
- `--pos_weight`: Positive class weight for imbalanced data (default: 2)
- `--mode`: Data mode - `pu` (protein+structure), `seq` (sequence only), or `str` (structure only)

## Evaluation and Inference

```bash
# Evaluate trained model
exp/prismnet/eval.sh <PROTEIN_NAME> <DATA_DIR>

# Inference on new data (TSV format)
exp/prismnet/infer.sh <PROTEIN_NAME> /path/to/inference_file.tsv

# Compute saliency maps
exp/prismnet/saliency.sh <PROTEIN_NAME> /path/to/inference_file.tsv

# Generate saliency visualizations (PDF images)
exp/prismnet/saliencyimg.sh <PROTEIN_NAME> /path/to/inference_file.tsv

# Compute high attention regions (20nt windows)
exp/prismnet/har.sh <PROTEIN_NAME> /path/to/inference_file.tsv
```

Output files are saved in `exp/<EXP_NAME>/out/`:
- Models: `out/models/<identity>_best.pth`
- Evaluations: `out/evals/<identity>.metrics` and `.probs`
- Inference: `out/infer/<identity>.probs`
- Saliency: `out/saliency/<identity>.sal`
- Visualizations: `out/imgs/<identity>/`

## Pre-trained Models

**IMPORTANT**: All 172 pre-trained PrismNet models are available in the parent project:

```
/home/shigo-45/projects/PrismNet/exp/train_all/out/models/
```

**When to use these models:**
- Evaluation tasks requiring trained PrismNet models
- Inference on new data
- Comparison with baseline models
- Analysis requiring model weights
- Any task mentioning "PrismNet model" or "trained model"

**Model naming convention:**
```
<PROTEIN>_<CELLTYPE>_PrismNet_pu_best.pth
```

Examples:
- `AARS_K562_PrismNet_pu_best.pth`
- `TIA1_Hela_PrismNet_pu_best.pth`
- `YTHDF2_Hela_PrismNet_pu_best.pth`

**Before training new models**, always check if a pre-trained model already exists in this directory. There are 172 models available covering all RBP datasets.

**Evaluation results** are also available:
```
/home/shigo-45/projects/PrismNet/exp/train_all/out/evals/
```

Format: `<PROTEIN>_<CELLTYPE>_PrismNet_pu.metrics` (TSV format: dataset, acc, auc, prc, tp, tn, fp, fn)

## Architecture

### Model Structure

PrismNet is a hybrid 2D/1D CNN with attention mechanisms:

1. **Input Processing**: RNA sequences (4 channels: ACGU) + icSHAPE structure (1 channel) → Shape: (batch, 1, 101, 5)
2. **2D Convolution**: Initial feature extraction with 11×5 kernels
3. **SE Block**: Squeeze-and-excitation attention mechanism
4. **ResidualBlock2D**: 2D residual blocks for spatial feature learning
5. **Pooling**: Average pooling across feature dimension
6. **ResidualBlock1D**: 1D residual blocks for sequential processing
7. **Global Pooling + FC**: Final classification layer

Two model variants:
- `PrismNet`: Base model (8 base channels)
- `PrismNet_large`: Larger model (64 base channels)

### Data Modes

The model supports three input modes (controlled via `--mode`):
- **`pu` (protein+structure)**: Uses all 5 features (ACGU + icSHAPE). Default mode.
- **`seq`**: Sequence only (4 channels, ACGU)
- **`str`**: Structure only (1 channel, icSHAPE)

Mode affects input slicing in forward pass and kernel dimensions.

### Key Components

- **Loader** ([prismnet/loader.py](prismnet/loader.py)): `SeqicSHAPE` dataset class handles HDF5 data and TSV inference files
- **Training Loop** ([prismnet/engine/train_loop.py](prismnet/engine/train_loop.py)): `train()`, `validate()`, `inference()` functions
- **Model** ([prismnet/model/PrismNet.py](prismnet/model/PrismNet.py)): Main architecture with SE blocks and residual blocks
- **Saliency** ([prismnet/model/smoothgrad.py](prismnet/model/smoothgrad.py)): GuidedBackpropSmoothGrad for interpretability
- **Main Script** ([tools/main.py](tools/main.py)): Entry point that orchestrates training/eval/inference

### Directory Structure

```
prismnet/
├── model/          # Neural network architectures
│   ├── PrismNet.py    # Main model
│   ├── resnet.py      # Residual blocks (1D and 2D)
│   ├── se.py          # Squeeze-excitation block
│   └── smoothgrad.py  # Saliency computation
├── engine/         # Training and evaluation loops
├── utils/          # Utilities (metrics, data processing, visualization)
└── loader.py       # Data loading

exp/                # Experiment configurations (shell scripts)
├── prismnet/       # Main experiments
├── train_one/      # Single protein training
└── train_all/      # Batch training

tools/
├── main.py         # Main entry point for all operations
└── generate_dataset.py  # Data preprocessing

data/               # Datasets (HDF5 and TSV files)
motif_construct/    # Perl/R scripts for motif analysis
```

## Important Implementation Details

- **Random Seed**: Fixed via `fix_seed()` for reproducibility (default: 1024)
- **Loss Function**: BCEWithLogitsLoss with configurable positive class weighting
- **Optimizer**: Adam with warmup scheduler (GradualWarmupScheduler, 8× multiplier)
- **Gradient Clipping**: Max norm of 5 applied during training
- **Early Stopping**: Based on validation AUC, with configurable patience
- **Data Format**: Input sequences are 101 nucleotides long
- **Model Checkpointing**: Best model saved based on validation AUC

## Experiment Workflow Pattern

1. Ensure data is in HDF5 format (existing TSV in `data/clip_data/` → convert with `tools/gdata_bin.sh`)
2. Create/modify experiment directory in `exp/` (copy from existing template)
3. Edit shell script parameters as needed
4. Run training script: `exp/<name>/train.sh <protein> <data_dir>`
5. Models save to `exp/<name>/out/models/`
6. Run evaluation: `exp/<name>/eval.sh <protein> <data_dir>`
7. Use inference scripts for new predictions

## Motif Construction

Separate workflow using Perl and R scripts in [motif_construct/](motif_construct/):

```bash
perl saliency_motif.pl infile.txt sal outfile
Rscript motif_sig.R outfile_motif_summary.txt outfile_motif_sig.txt
```

## Half Life Analysis Example

Additional analysis pipeline requiring extra dependencies:

```bash
# Install dependencies
pip install xgboost==1.3.0rc1 matplotlib scipy scikit-learn termplotlib

# Run analysis (data should be in data/halflife_data/)
exp/logistic_reg/run.sh
```

## Reference

Project website: http://prismnet.zhanglab.net/

When working with this codebase, note that the primary workflow is through bash scripts in `exp/` rather than direct Python calls. All configuration happens via command-line arguments passed through these scripts.
