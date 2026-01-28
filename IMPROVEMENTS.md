# PrismNet_Remastered

This document outlines potential improvements to the PrismNet architecture and codebase based on modern deep learning practices (as of 2026).

## 1. Architecture Modernization

### Replace CNNs with Transformer-based Models
- **Current State**: Hybrid 2D/1D CNN with SE attention blocks
- **Proposed Improvement**: Use Transformers or hybrid CNN-Transformer architectures
  - Models like DNABERT, Nucleotide Transformer, or Enformer have shown superior performance on genomic tasks
  - Self-attention can capture long-range dependencies better than CNNs
  - Could still use CNN for local feature extraction + Transformer for global context
- **Implementation Notes**:
  - Consider using pre-trained models from HuggingFace Transformers library
  - May need to adapt input format for transformer models
- **Priority**: High
- **Estimated Complexity**: High

### More Sophisticated Attention Mechanisms
- **Current State**: Only SE (Squeeze-and-Excitation) blocks for channel attention
- **Proposed Improvements**:
  - Multi-head self-attention to model position-position interactions
  - Cross-attention between sequence and structure modalities
  - CBAM (Convolutional Block Attention Module) for both channel and spatial attention
- **Files to Modify**:
  - `prismnet/model/PrismNet.py`
  - Create new attention modules in `prismnet/model/`
- **Priority**: Medium
- **Estimated Complexity**: Medium

## 2. Input Handling

### Variable-Length Sequences
- **Current State**: Fixed 101nt window size
- **Proposed Improvement**: Support variable-length inputs using:
  - Positional encodings + padding/masking
  - Adaptive pooling layers
  - Allows processing of full transcripts or longer contexts
- **Files to Modify**:
  - `prismnet/loader.py` - Update SeqicSHAPE dataset class
  - `prismnet/model/PrismNet.py` - Add dynamic padding/masking support
- **Priority**: Medium
- **Estimated Complexity**: Medium

### Better Multi-Modal Fusion
- **Current State**: Simple concatenation of sequence + structure (5 channels)
- **Proposed Improvements**:
  - Late fusion: Process sequence and structure in separate encoders, then combine
  - Cross-modal attention: Let sequence features attend to structure features and vice versa
  - Gated fusion: Learn when to rely on sequence vs. structure
- **Implementation Notes**:
  - Create separate encoder branches in model architecture
  - Add fusion layer before final classification
- **Priority**: High
- **Estimated Complexity**: High

## 3. Training Enhancements

### Modern Optimizers and Schedulers
- **Current State**: Adam optimizer with warmup scheduler
- **Proposed Improvements**:
  - AdamW (weight decay fix)
  - Cosine annealing with warm restarts
  - OneCycleLR for faster convergence
- **Files to Modify**:
  - `tools/main.py` - Update optimizer initialization
  - `prismnet/engine/train_loop.py` - Update training loop
- **Priority**: Low (easy win)
- **Estimated Complexity**: Low

### Data Augmentation
- **Current State**: No apparent augmentation in data loading
- **Proposed Improvements**:
  - Random shifts/crops within sequences
  - Masking strategies (like BERT's masked language modeling)
  - Structure noise injection to improve robustness
  - Reverse complement augmentation
- **Files to Modify**:
  - `prismnet/loader.py` - Add augmentation transforms
  - Create new `prismnet/utils/augmentation.py` module
- **Priority**: Medium (good ROI)
- **Estimated Complexity**: Low-Medium

### Mixed Precision Training
- **Proposed Improvement**: Use `torch.cuda.amp` for faster training and lower memory usage
- **Files to Modify**:
  - `prismnet/engine/train_loop.py` - Add GradScaler and autocast
  - `tools/main.py` - Add command-line flag for mixed precision
- **Priority**: Low (performance optimization)
- **Estimated Complexity**: Low

## 4. Transfer Learning & Pre-training

### Pre-trained Foundation Models
- **Current State**: Trains from scratch for each protein
- **Proposed Improvements**:
  - Pre-train on large unlabeled RNA datasets (icSHAPE data + RNA sequences)
  - Fine-tune for specific protein binding tasks
  - Use existing pre-trained models like RNA-FM or DNABERT-2
- **Implementation Notes**:
  - Add pre-training script in `tools/`
  - Create adapter layers for fine-tuning
- **Priority**: High (major performance boost potential)
- **Estimated Complexity**: Very High

### Multi-Task Learning
- **Current State**: Each protein trained separately (172 separate models)
- **Proposed Improvement**:
  - Single multi-task model predicting binding for all 172 proteins simultaneously
  - Shared encoder + protein-specific heads
  - Better generalization and fewer parameters overall
- **Files to Modify**:
  - `prismnet/model/PrismNet.py` - Add multi-head output layer
  - `prismnet/loader.py` - Update to load data for all proteins
  - `prismnet/engine/train_loop.py` - Update loss computation for multi-task
  - `exp/prismnet/train_multitask.sh` - New training script
- **Priority**: High
- **Estimated Complexity**: High

## 5. Addressing Class Imbalance

### Better Loss Functions
- **Current State**: BCEWithLogitsLoss with fixed `pos_weight=2`
- **Proposed Improvements**:
  - Focal Loss to focus on hard examples
  - Dynamic class weighting based on actual data distribution
  - AUC maximization loss since AUC is the evaluation metric
- **Files to Modify**:
  - Create `prismnet/utils/losses.py` with custom loss functions
  - `prismnet/engine/train_loop.py` - Use new loss functions
  - `tools/main.py` - Add command-line argument for loss type
- **Priority**: Medium-High
- **Estimated Complexity**: Low-Medium

## 6. Interpretability & Explainability

### Advanced Saliency Methods
- **Current State**: GuidedBackpropSmoothGrad
- **Proposed Improvements**:
  - Integrated Gradients (more stable attributions)
  - Attention rollout if using Transformers
  - SHAP values for model-agnostic explanations
  - Layer-wise relevance propagation (LRP)
- **Files to Modify**:
  - `prismnet/model/smoothgrad.py` - Add new saliency methods
  - `exp/prismnet/saliency.sh` - Update to support multiple methods
- **Priority**: Medium
- **Estimated Complexity**: Medium

## 7. Model Efficiency

### Reduce Model Size
- **Current State**: Two variants (base: 8 channels, large: 64 channels)
- **Proposed Improvements**:
  - Knowledge distillation: Train smaller student from larger teacher
  - Pruning & quantization for deployment
  - Depthwise separable convolutions to reduce parameters
- **Implementation Notes**:
  - Add `tools/distill.py` for knowledge distillation
  - Use PyTorch quantization tools
- **Priority**: Low (optimization, not core functionality)
- **Estimated Complexity**: Medium

### Efficient Attention
- **Proposed Improvement**: For longer sequences, use efficient attention variants:
  - Linear attention, Performer, or Flash Attention
- **Files to Modify**:
  - Add efficient attention implementations in `prismnet/model/`
- **Priority**: Low (only needed if supporting longer sequences)
- **Estimated Complexity**: Medium

## 8. Code & Infrastructure

### Modern ML Ops
- **Current State**: Custom training loops, TensorBoard logging, shell scripts for configuration
- **Proposed Improvements**:
  - PyTorch Lightning for cleaner code and built-in features
  - Weights & Biases (wandb) for better experiment tracking
  - Hydra for configuration management instead of shell scripts
  - Ray Tune for hyperparameter optimization
- **Files to Modify**:
  - Refactor `prismnet/engine/train_loop.py` to use Lightning
  - Replace shell scripts in `exp/` with Hydra configs
  - Add `configs/` directory for YAML configurations
- **Priority**: Medium (improves maintainability)
- **Estimated Complexity**: High (major refactoring)

### Enhanced Reproducibility
- **Current State**: Has `fix_seed()` but manual seed setting in scripts
- **Proposed Improvements**:
  - Deterministic CUDA operations
  - Better logging of environment (package versions, hardware)
  - Docker containers for full reproducibility
- **Files to Modify**:
  - Add `Dockerfile` and `docker-compose.yml`
  - Create `environment.yml` or `poetry.lock` for exact dependencies
  - Update `prismnet/utils/misc.py` with deterministic settings
- **Priority**: Low-Medium
- **Estimated Complexity**: Low

## 9. Evaluation & Benchmarking

### Cross-Validation
- **Current State**: Single train/val/test split
- **Proposed Improvement**: K-fold cross-validation for more robust performance estimates
- **Files to Modify**:
  - `prismnet/loader.py` - Add k-fold split functionality
  - `tools/main.py` - Add cross-validation mode
  - Create `exp/prismnet/train_cv.sh` for cross-validation
- **Priority**: Low-Medium
- **Estimated Complexity**: Low

### Additional Metrics
- **Current State**: Primarily AUC
- **Proposed Improvements**:
  - Precision-Recall curves and AUPRC (better for imbalanced data)
  - Calibration metrics (Brier score, calibration plots)
  - Per-protein performance analysis
- **Files to Modify**:
  - `prismnet/utils/metrics.py` - Add new metric calculations
  - `prismnet/engine/train_loop.py` - Log additional metrics
- **Priority**: Medium
- **Estimated Complexity**: Low

## 10. Biological Priors

### Structure-Aware Architectures
- **Proposed Improvements**:
  - Incorporate RNA secondary structure predictions (dot-bracket notation)
  - Graph neural networks (GNNs) to model base-pairing interactions
  - Convolutional kernels that respect RNA structural motifs
- **Implementation Notes**:
  - Add GNN layers using PyTorch Geometric
  - Precompute secondary structures using tools like RNAfold
- **Priority**: High (biological relevance)
- **Estimated Complexity**: Very High

### Evolutionary Information
- **Proposed Improvement**: Add evolutionary conservation scores or sequence co-evolution features
- **Implementation Notes**:
  - Integrate with databases like PhyloP or PhastCons
  - Add as additional input channels
- **Files to Modify**:
  - `prismnet/loader.py` - Load conservation data
  - `tools/generate_dataset.py` - Preprocess conservation scores
- **Priority**: Medium
- **Estimated Complexity**: Medium

---

## Priority Implementation Plan

### Phase 1: Quick Wins (Low complexity, good ROI)
1. **Modern optimizers** (AdamW, better schedulers)
2. **Data augmentation** (random shifts, masking, reverse complement)
3. **Additional evaluation metrics** (AUPRC, calibration)
4. **Better loss functions** (Focal Loss)

### Phase 2: Core Improvements (High impact)
5. **Multi-task learning** across all proteins
6. **Better multi-modal fusion** (separate encoders for sequence/structure)
7. **Transfer learning** with pre-trained models
8. **Transformer architecture** (or hybrid CNN-Transformer)

### Phase 3: Infrastructure (Long-term maintainability)
9. **PyTorch Lightning** refactoring
10. **Hydra configuration** management
11. **Docker containerization**
12. **Comprehensive logging** (wandb)

### Phase 4: Advanced Features (Research extensions)
13. **Variable-length sequences**
14. **Graph neural networks** for structure
15. **Knowledge distillation** for efficiency
16. **Advanced interpretability** methods

---

## Implementation Checklist

- [ ] Create feature branches for each major improvement
- [ ] Write unit tests for new functionality
- [ ] Benchmark performance before/after changes
- [ ] Update documentation and CLAUDE.md
- [ ] Run experiments on subset of proteins before full training
- [ ] Compare with baseline model performance

---

## References & Resources

- **Transformers for Genomics**:
  - DNABERT: https://github.com/jerryji1993/DNABERT
  - Nucleotide Transformer: https://github.com/instadeepai/nucleotide-transformer
  - Enformer: https://github.com/deepmind/deepmind-research/tree/master/enformer

- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/stable/
- **Hydra**: https://hydra.cc/
- **Weights & Biases**: https://wandb.ai/
- **PyTorch Geometric** (for GNNs): https://pytorch-geometric.readthedocs.io/

---

## Notes

- All improvements should maintain backward compatibility with existing trained models where possible
- Performance comparisons should use the same train/val/test splits as original paper
- Consider computational cost vs. performance gain for each improvement
- Prioritize improvements that enhance biological interpretability

---

**Last Updated**: 2026-01-27
**Document Version**: 1.0
