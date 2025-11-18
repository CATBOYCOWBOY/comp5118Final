# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simplified test bench for analyzing LLM performance on NL2SQL tasks using the Spider dataset. The project uses a multi-stage prompting strategy with Novita AI's LLM service and supports ablation studies to understand the impact of different pipeline components.

## Project Structure

- **`spider/`** - Contains the Spider dataset and evaluation framework (gitignored but contains core evaluation logic)
  - `evaluation.py` - Main evaluation script for SQL accuracy metrics
  - `process_sql.py` - SQL parsing and processing utilities
- **`src/`** - Simplified source code directory:
  - `testbench.py` - Main orchestrator with simple dictionary-based configs
  - `novita_client.py` - Novita AI integration
  - `spider.py` - Merged Spider dataset loading and evaluation
  - `strategies.py` - Multi-stage prompting strategy with ablation support
  - `performance.py` - Merged performance tracking and visualization using JSON
  - `config.py` - Simple configuration with hardcoded model lists
- **`results/`** - Output directory for JSON-based experiment results
- **`secrets.txt`** - API keys and configuration (gitignored)
- **`run_testbench.py`** - Simplified CLI interface

## Available Models

The system supports these hardcoded models (defined in `src/config.py`):
- `meta-llama/llama-3.2-3b-instruct`
- `meta-llama/llama-3.2-11b-vision-instruct`
- `deepseek/deepseek-r1`
- `qwen/qwen-2.5-72b-instruct`

## Strategy Variants (Ablation Support)

The multi-stage strategy supports ablation studies with these variants:
- `multi_stage` - Full pipeline: analysis → generation → verification
- `multi_stage_no_analysis` - Skip analysis step
- `multi_stage_no_verification` - Skip verification step
- `multi_stage_simple` - Generation only

## Quick Usage

### Command Line Interface
```bash
# Quick test
python run_testbench.py quick --model meta-llama/llama-3.2-3b-instruct --examples 10

# Compare all models
python run_testbench.py compare-models --examples 50

# Compare strategies (ablation study)
python run_testbench.py compare-strategies --model meta-llama/llama-3.2-3b-instruct

# List available options
python run_testbench.py list-models
python run_testbench.py list-strategies
```

### Python API
```python
from src.testbench import NL2SQLTestbench

# Quick test
testbench = NL2SQLTestbench()
await testbench.quick_test(model_name="meta-llama/llama-3.2-3b-instruct", num_examples=10)

# Strategy ablation
await testbench.compare_strategies(
    strategies=["multi_stage", "multi_stage_no_verification"],
    num_examples=50
)
```

## Configuration System

The system uses simple Python dictionaries instead of complex objects:

```python
# Example config
{
    "name": "experiment_name",
    "models": ["meta-llama/llama-3.2-3b-instruct"],
    "strategy": "multi_stage",
    "max_examples": 50,
    "spider_path": "spider",
    "split": "dev"
}
```

## Performance Tracking

Results are stored as simple JSON with this structure:
```json
{
    "timestamp": "2024-01-01T12:00:00",
    "model_name": "meta-llama/llama-3.2-3b-instruct",
    "strategy": "multi_stage",
    "accuracy": {
        "exact_match": true,
        "execution_match": true
    },
    "performance": {
        "response_time": 2.5,
        "tokens_used": 150
    }
}
```

## Key Simplifications

- **No Complex Dataclasses**: Uses simple Python dictionaries
- **No CSV Files**: All data stored as JSON
- **Merged Classes**: Combined related functionality
- **Hardcoded Lists**: Models and strategies defined in config
- **Simple CLI**: Direct function calls, no complex object setup
- **Multi-Stage Only**: Focused on one effective strategy with ablation variants

## Development Notes

- The Spider directory is gitignored but contains essential evaluation infrastructure
- API credentials should be stored in `secrets.txt` (gitignored)
- The project focuses on multi-stage prompting ablation studies
- All performance data is JSON-based for easy analysis
- System designed for simplicity and ease of modification