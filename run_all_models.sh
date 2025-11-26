#!/bin/bash

# Simple script to run all Spider models with full dataset

echo "Starting full Spider evaluation on all models..."

models=(
    "qwen/qwen2.5-7b-instruct"
    "openai/gpt-oss-20b"
)

for model in "${models[@]}"; do
    echo "=========================================="
    echo "Running model: $model"
    echo "=========================================="
    python3 run_testbench.py run_test --model "$model" --examples 1034
    echo ""
done

echo "All evaluations complete!"
