# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **NL2SQL Testbench** for evaluating Text-to-SQL predictions on the Spider dataset using various large language models via Novita AI. The system generates SQL queries from natural language questions and evaluates them using test-suite-sql-eval for execution accuracy.

## Key Commands

### Environment Setup
```bash
./startup.sh                    # Bootstrap virtualenv and install dependencies
source .venv/bin/activate       # Activate the virtual environment
```

### Core Operations
```bash
# Single model test (basic)
python run_testbench.py run_test --model meta-llama/llama-3.2-3b-instruct --examples 20

# Single model test with evaluation
python run_testbench.py run_test --model meta-llama/llama-3.2-3b-instruct --examples 20 --evaluate

# Re-evaluate existing results
python run_testbench.py evaluate_run --run-dir results/<test_directory>

# Show dataset information
python run_testbench.py info

# List available models
python run_testbench.py list-models

# Batch run all models (long-running)
./run_all_models.sh
```

### Testing and Development
```bash
# Quick smoke test (5 examples)
python run_testbench.py run_test --model meta-llama/llama-3.2-3b-instruct --examples 5

# Compare multiple models (from AGENTS.md)
python run_testbench.py compare-models --models meta-llama/llama-3.2-3b-instruct qwen/qwen3-coder-30b-a3b-instruct --examples 20
```

## Architecture Overview

### Core Components
- **`run_testbench.py`**: Main CLI entry point with argument parsing and command dispatch
- **`src/gathering/spider1_testbench.py`**: Core orchestrator handling test execution, result collection, and evaluation integration
- **`src/gathering/spider1.py`**: Dataset loading, Spider format handling, and test-suite-sql-eval integration
- **`src/gathering/prompting_strategy.py`**: Prompt generation and SQL extraction logic
- **`src/gathering/novita_client.py`**: Async API client for LLM inference with retry/backoff logic
- **`src/gathering/config.py`**: Model definitions and configuration management

### Data Flow
1. **Dataset Loading**: `spider1.py` loads Spider dev.json and database schemas
2. **Prompt Generation**: `prompting_strategy.py` creates structured prompts with schema context
3. **LLM Inference**: `novita_client.py` makes async API calls with retry logic
4. **SQL Extraction**: Post-processing extracts SQL from model responses
5. **Evaluation**: `spider1.py` integrates with test-suite-sql-eval for execution accuracy
6. **Results Storage**: Structured outputs saved to `results/<run_name>/`

### External Dependencies
- **`spider/`**: Spider dataset (dev.json, database files, table schemas)
- **`test-suite-sql-eval/`**: Upstream evaluation framework for SQL execution accuracy
- **Novita AI API**: External service for LLM inference (requires API key)

## Configuration

### API Keys
- Required: `NOVITA_API_KEY` in environment or `NOVITA_API_KEY` file
- Security: API keys are gitignored, never commit credentials

### Model Management
- Available models hardcoded in `src/gathering/config.py:AVAILABLE_MODELS`
- Remove unavailable models to avoid 404 errors during batch runs
- Default parameters: temperature=0.1, max_tokens=1024, retry_count=3

### Output Structure
Each run creates `results/<name>_<timestamp>/`:
- `<model>_<strategy>/predictions.txt`: Generated SQL queries (tab-separated with db_id)
- `<model>_<strategy>/gold.txt`: Gold standard SQL queries
- `<model>_<strategy>/evaluation_output.txt`: test-suite-sql-eval results
- `results.json`: Aggregated metrics and run metadata
- `metadata.json`: Per-model/strategy configuration

## Development Guidelines

### Code Style
- Python with PEP 8: 4-space indentation, snake_case, type hints preferred
- Async/await patterns: Maintain async flows for Novita API calls
- Configuration over hardcoding: Use `config.py` or CLI args, avoid magic values

### Testing Approach
- Use small example counts (5-10) for regression testing
- Run `compare-models` when touching prompting or evaluation logic
- Verify result artifacts are generated and parsable
- No formal unit test suite - rely on targeted integration tests

### Security Considerations
- API keys via environment or gitignored files only
- Keep example counts low in development to avoid unnecessary API costs
- Async API client includes exponential backoff and rate limiting