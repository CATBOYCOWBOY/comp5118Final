# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Spider1 testbench for analyzing LLM performance on traditional NL2SQL tasks using the Spider dataset. The project implements test-suite-sql-eval evaluation with Novita AI's LLM service and provides comprehensive benchmarking of models on SQL generation accuracy.

## Project Structure

- **`Spider2/`** - Contains the Spider2-lite dataset and SQLite databases
  - `spider2-lite/` - Spider2-lite dataset with 135 local examples
  - `spider2-lite/resource/databases/local_sqlite/` - 30 real SQLite databases
  - `spider2-lite/resource/documents/` - External knowledge documents
- **`src/`** - Source code directory:
  - `spider2_testbench.py` - Main Spider2 testbench orchestrator
  - `novita_client.py` - Novita AI integration
  - `spider2.py` - Spider2-lite dataset manager and evaluation
  - `strategies.py` - Spider2 prompting strategy
  - `performance.py` - Performance tracking and visualization
  - `config.py` - Configuration with hardcoded model lists
- **`results/`** - Output directory for JSON-based experiment results
- **`NOVITA_API_KEY`** - API key file (gitignored)
- **`run_testbench.py`** - Spider2-lite CLI interface

## Available Models

The system supports these hardcoded models (defined in `src/config.py`):
- `meta-llama/llama-3.2-3b-instruct`
- `meta-llama/llama-3.1-8b-instruct`
- `meta-llama/llama-3.2-11b-vision-instruct`
- `qwen/qwen-2.5-14b-instruct`
- `qwen/qwen3-coder-30b-a3b-instruct`
- `google/gemma-3-12b-it`
- `google/gemma-3-27b-it`
- `mistralai/mistral-7b-instruct-v0.3`
- `deepseek/deepseek-r1-0528-qwen3-8b`

## Strategy Configuration

The system uses a single comprehensive strategy:
- `spider2_basic` - Text-in-results-out evaluation with schema context

## Quick Usage

### Command Line Interface
```bash
# Quick test
python run_testbench.py quick --model qwen/qwen3-coder-30b-a3b-instruct --examples 10

# Compare models
python run_testbench.py compare-models --examples 50

# Show dataset info
python run_testbench.py info

# List available models
python run_testbench.py list-models
```

### Python API
```python
from src.spider2_testbench import Spider2Testbench

# Quick test
testbench = Spider2Testbench()
await testbench.quick_test(model_name="qwen/qwen3-coder-30b-a3b-instruct", max_examples=10)

# Model comparison
await testbench.compare_models(max_examples=50)
```

## Configuration System

The system uses simple Python dictionaries:

```python
# Example config
{
    "name": "spider2_quick_test",
    "models": ["qwen/qwen3-coder-30b-a3b-instruct"],
    "strategy": "spider2_basic",
    "max_examples": 10,
    "spider2_path": "Spider2",
    "dataset_type": "spider2"
}
```

## Performance Tracking

Results are stored as JSON with this structure:
```json
{
    "timestamp": "2025-11-19T12:54:10",
    "dataset_type": "spider2-lite",
    "experiments": {
        "config": {...},
        "experiments": {
            "model_strategy": {
                "summary": {
                    "total_examples": 10,
                    "execution_accuracy": 0.8,
                    "successful_examples": 8
                },
                "entries": [...]
            }
        }
    }
}
```

## Key Features

- **Text-in-Results-Out**: Evaluation based on SQL execution results, not string matching
- **Real Enterprise Data**: 30 SQLite databases with complex schemas from real businesses
- **Complex Analytics**: Time series, RFM analysis, window functions, CTEs
- **External Knowledge**: Domain-specific documentation integration
- **Execution-Based**: True semantic evaluation through SQL execution

## Spider2-lite Dataset

- **135 local examples** across 30 different databases
- **Complex business questions** requiring advanced SQL
- **Real-world schemas** from e-commerce, sports, entertainment, finance
- **External knowledge documents** for domain-specific analysis
- **SQLite format** for local evaluation without cloud costs

## Key Improvements from Spider1

- **Enterprise Focus**: Real-world business analytics vs academic examples
- **Execution Evaluation**: Results comparison vs regex string matching
- **Complex Queries**: Multi-table joins, window functions, CTEs
- **Domain Knowledge**: Business context integration
- **Scalable Evaluation**: Local SQLite databases vs cloud dependencies

## Development Notes

- All experiments automatically save results to `results/` with timestamps
- The system focuses on execution accuracy as the primary metric
- External knowledge documents provide business context for complex queries
- SQL extraction handles multiple response formats from different LLMs
- Database schemas are automatically introspected and formatted for prompts

# Important Instructions
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.