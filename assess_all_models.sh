#!/bin/bash

# Simple script to assess all Spider models with full dataset after test run

echo "Starting full Spider evaluation on all models..."

test_dirs=(
    "results/gemma_3_27b_it_test_20251125_044115"
    "results/qwen3_coder_30b_a3b_instruct_test_20251125_052712"
    "results/qwen3_32b_fp8_test_20251125_063123"
)

for test_dir in "${test_dirs[@]}"; do
    echo "=========================================="
    echo "Asessing test: $test_dir"
    echo "=========================================="
    python run_testbench.py evaluate_run --run-dir "$test_dir"
    echo ""
done

echo "All evaluations complete!"
