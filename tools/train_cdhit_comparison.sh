#!/bin/bash
# Train PrismNet on both original and CD-HIT splits for comparison
# This validates that homology-aware splitting doesn't degrade performance

set -e

echo "=========================================="
echo "PrismNet CD-HIT Split Validation"
echo "=========================================="
echo ""

# Datasets to test
DATASETS=("TIA1_Hela" "IGF2BP1_K562" "SRSF1_HepG2")

# Create output directory for comparison
COMPARISON_DIR="evaluation/cdhit_validation"
mkdir -p $COMPARISON_DIR

echo "Training 3 datasets with both original and CD-HIT splits..."
echo "This will take approximately 2-3 hours total."
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Dataset: $dataset"
    echo "=========================================="
    echo ""

    # Train on original split
    echo "1. Training on ORIGINAL split..."
    cd /home/shigo-45/projects/PrismNet
    exp/prismnet/train.sh $dataset clip_data

    # Save results
    mkdir -p $COMPARISON_DIR/${dataset}_original
    cp -r exp/prismnet/out/* $COMPARISON_DIR/${dataset}_original/

    # Clean up for next run
    rm -rf exp/prismnet/out

    echo ""
    echo "2. Training on CD-HIT split..."

    # Train on CD-HIT split
    exp/prismnet/train.sh ${dataset}_cdhit80 clip_data

    # Save results
    mkdir -p $COMPARISON_DIR/${dataset}_cdhit80
    cp -r exp/prismnet/out/* $COMPARISON_DIR/${dataset}_cdhit80/

    # Clean up for next run
    rm -rf exp/prismnet/out

    echo ""
    echo "✓ Completed $dataset"
    echo ""
done

echo "=========================================="
echo "All training complete!"
echo "=========================================="
echo ""
echo "Results saved to: $COMPARISON_DIR"
echo ""
echo "Next: Run comparison analysis"
