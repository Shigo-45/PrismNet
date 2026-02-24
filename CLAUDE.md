# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PrismNet is a PyTorch-based deep learning model for predicting dynamic cellular protein-RNA interactions using in vivo RNA structure data (icSHAPE). The model uses a CNN architecture with residual blocks and squeeze-and-excitation modules to process sequence and structure features.

## Package Management

This project uses **uv** for dependency management. The package is configured via `pyproject.toml`:
- Install dependencies: `uv sync`
- Add new package: `uv add <package-name>`

## Project Structure

### Core Package (`prismnet/`)
- `model/PrismNet.py` - Main CNN architecture with three modes: "pu" (sequence+structure), "seq" (sequence only), "str" (structure only)
- `model/resnet.py` - 1D and 2D residual blocks
- `model/se.py` - Squeeze-and-Excitation attention modules
- `model/smoothgrad.py` - SmoothGrad for model interpretability
- `engine/train_loop.py` - Training and validation loops
- `loader.py` - Data loader for h5py datasets and TSV inference files
- `utils/` - Metrics, data utilities, visualization

### Entry Point (`tools/`)
- `main.py` - Main script that handles training, evaluation, inference, saliency computation
- Other dataset generation scripts

### Experiments (`exp/`)
Each experiment directory (e.g., `exp/prismnet/`, `exp/train_all/`) contains shell scripts for different operations. Scripts expect to be run from the experiment directory and use relative paths.

## Common Commands

### Training
Train a single protein model:
```bash
exp/prismnet/train.sh <PROTEIN_NAME> <DATA_DIR>
# Example: exp/prismnet/train.sh TIA1_Hela clip_data
```

Train all protein models:
```bash
exp/prismnet/train_all.sh
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
exp/prismnet/eval.sh <PROTEIN_NAME> <DATA_DIR>
# Example: exp/prismnet/eval.sh TIA1_Hela clip_data
```

### Inference
Run inference on new data (TSV format):
```bash
exp/prismnet/infer.sh <PROTEIN_NAME> <INFERENCE_FILE>
# Example: exp/prismnet/infer.sh TIA1_Hela /path/to/data.tsv
```

### Interpretability
Compute saliency maps:
```bash
exp/prismnet/saliency.sh <PROTEIN_NAME> <INFERENCE_FILE>
```

Generate saliency visualizations:
```bash
exp/prismnet/saliencyimg.sh <PROTEIN_NAME> <INFERENCE_FILE>
```

Compute high attention regions:
```bash
exp/prismnet/har.sh <PROTEIN_NAME> <INFERENCE_FILE>
```

### TensorBoard Monitoring
To monitor training, add `-tfboard` flag in training scripts and run:
```bash
tensorboard --logdir exp/<EXP_NAME>/out/tfb
```

## Data Format

### Training Data
- Stored as HDF5 files with keys: `X_train`, `Y_train`, `X_test`, `Y_test`
- Input shape: `(batch, 1, seq_length, n_features)` where n_features=5 for pu mode (4 nucleotides + 1 structure)

### Inference Data
- TSV files with columns for sequence and optional structure features
- Loaded via `datautils.load_testset_txt()` in `prismnet/utils/datautils.py`

## Architecture Notes

### PrismNet Model Flow
1. **Input**: (batch, 1, seq_length, n_features)
2. **Conv2d**: Initial 2D convolution with batch norm
3. **SEBlock**: Squeeze-and-excitation attention
4. **ResidualBlock2D**: 2D residual learning
5. **AvgPool2d**: Pool across feature dimension
6. **ResidualBlock1D**: 1D residual learning on sequence
7. **Global pooling + FC**: Final classification

### Training Loop (`tools/main.py`)
- Uses `GradualWarmupScheduler` for learning rate warmup
- Supports both binary classification and regression modes
- Gradient clipping at norm 5
- Skips batches with all positive or all negative samples
- Saves best model based on validation AUC

### Output Structure
When running experiments, outputs are saved to `exp/<EXP_NAME>/out/`:
- `ckpt/` - Model checkpoints
- `evals/` - Evaluation metrics and probabilities
- `tfb/` - TensorBoard logs
- `log.txt` - Training logs

## Experiment Workflow

1. Generate dataset from raw data: `tools/generate_dataset.py`
2. Train model using experiment scripts in `exp/*/train.sh`
3. Models are saved to `exp/<EXP_NAME>/out/ckpt/`
4. Evaluate using `eval.sh` (loads best model with `--load_best` flag)
5. Run inference/interpretability analysis on new data

## Git Workflow

- Main branch: `master`
- Current development branch: `replica-win`
- Standard workflow: feature branch → `dev` → `master`
