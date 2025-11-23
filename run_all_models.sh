#!/bin/bash

# Simple script to run all Spider models with full dataset

echo "Starting full Spider evaluation on all models..."

models=(
    "meta-llama/llama-3.2-3b-instruct"
    "meta-llama/llama-3.1-8b-instruct"
    "deepseek/deepseek-r1-0528-qwen3-8b"
    "google/gemma-3-12b-it"
    "deepseek/deepseek-r1-distill-qwen-14b"
    "google/gemma-3-27b-it"
    "qwen/qwen3-coder-30b-a3b-instruct"
    "qwen/qwen3-32b-fp8"
)

for model in "${models[@]}"; do
    echo "=========================================="
    echo "Running model: $model"
    echo "=========================================="
    python3 run_testbench.py quick --model "$model" --examples 999999
    echo ""
done

echo "All evaluations complete!"
