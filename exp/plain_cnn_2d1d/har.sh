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
