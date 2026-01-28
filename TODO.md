# Documentation Understanding Checklist

## Core Project Documentation
- [X] README.md - Main project overview, setup, and usage instructions
- [X] LICENSE - Project licensing terms
- [X] setup.py - Python package configuration and dependencies
- [X] requirements.txt - Python dependencies list

## PrismNet Core Package
### Package Structure
- [X] prismnet/__init__.py - Package initialization
- [X] prismnet/loader.py - Data loading functionality
- [ ] prismnet/engine/__init__.py - Engine package init
- [ ] prismnet/engine/train_loop.py - Training loop implementation
- [ ] prismnet/model/__init__.py - Model package init
- [ ] prismnet/model/PrismNet.py - Main PrismNet model architecture
- [ ] prismnet/model/resnet.py - ResNet implementation for PrismNet
- [ ] prismnet/model/se.py - Squeeze-and-Excitation modules
- [ ] prismnet/model/smoothgrad.py - SmoothGrad implementation for interpretability
- [ ] prismnet/model/utils.py - Model utility functions
- [ ] prismnet/utils/__init__.py - Utils package init
- [ ] prismnet/utils/datautils.py - Data processing utilities
- [ ] prismnet/utils/metrics.py - Evaluation metrics
- [ ] prismnet/utils/visualize.py - Visualization functions
- [ ] prismnet/utils/xprint.py - Extended printing utilities
- [ ] prismnet/utils/acgu.npz - ACGU data file (representative for data files)

## Tools and Scripts
- [ ] tools/gdata_bin.sh - Data generation binary script
- [ ] tools/generate_dataset.py - Dataset generation tool
- [ ] tools/main.py - Main tool entry point

## Motif Construction
- [ ] motif_construct/motif_sig.R - R script for motif significance analysis
- [ ] motif_construct/saliency_motif.pl - Perl script for saliency motif analysis

## Experiments scripts
- [ ] exp/logistic_reg/ - Logistic regression experiment 
  - [ ] gdata.py - Data generation for logistic regression
  - [ ] main.py - Main logistic regression implementation
  - [ ] run.sh - Execution script for logistic regression

- [X] exp/prismnet/train.sh - Training script for single protein
- [X] exp/prismnet/train_all.sh - Training script for all proteins
- [X] exp/prismnet/eval.sh - Evaluation script for trained models
- [X] exp/prismnet/infer.sh - Inference script for new data predictions
- [X] exp/prismnet/saliency.sh - Saliency map computation script
- [X] exp/prismnet/saliencyimg.sh - Saliency visualization generation script
- [X] exp/prismnet/saliencyimg_infer.sh - Saliency visualization for inference data
- [X] exp/prismnet/har.sh - High attention regions computation script

## Understanding Notes
- Focus on understanding the purpose and functionality of each component
- Pay special attention to data flow and model architecture
- Note any configuration parameters and their effects
- Understand the relationship between different modules
- Identify key algorithms and methodologies used

## Progress Tracking
- Total documents to review: ~35
- Documents reviewed: 0
- Understanding level: Not started