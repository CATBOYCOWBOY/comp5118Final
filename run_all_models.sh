#!/bin/bash

# Simple script to run all Spider models with full dataset

echo "Starting full Spider evaluation on all models..."

models=(
    "qwen/qwen3-8b-fp8"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "qwen/qwen3-coder-30b-a3b-instruct"
    "qwen/qwen3-32b-fp8"
)

for model in "${models[@]}"; do
    echo "=========================================="
    echo "Running model: $model"
    echo "=========================================="
    python3 run_testbench.py run_test --model "$model" --examples 1034
    echo ""
done

echo "All evaluations complete!"
