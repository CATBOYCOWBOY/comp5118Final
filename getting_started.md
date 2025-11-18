# Getting Started with NL2SQL Testbench

## Quick Setup

### Prerequisites

- Python 3.8+ (Python 3.12 recommended)
- Git
- Novita AI API key

### Automated Setup

1. **Clone and setup everything:**
   ```bash
   git clone <repository-url>
   cd comp5118/final
   ./startup.sh
   ```

2. **Add your API key:**
   ```bash
   echo "NOVITA_API_KEY=your_actual_api_key_here" > secrets.txt
   ```

3. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   ```

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone Spider dataset
git clone https://github.com/taoyds/spider.git

# Create directories
mkdir -p results plots configs

# Set up API key
echo "NOVITA_API_KEY=your_api_key_here" > secrets.txt
```

## Your First Test

### Quick Test (5 examples)
```bash
python run_testbench.py quick
```

### Compare Models
```bash
python run_testbench.py preset model_comparison
```

### Interactive Examples
```bash
python examples/quick_start.py
```

## Basic Usage Examples

### 1. Simple Python Script

```python
import asyncio
from src import NL2SQLTestbench

async def main():
    testbench = NL2SQLTestbench()

    results = await testbench.quick_test(
        model_name="meta-llama/llama-3.2-3b-instruct",
        strategy="few_shot",
        num_examples=10
    )

    print(f"Accuracy: {results['accuracy']}")
    await testbench.close()

asyncio.run(main())
```

### 2. Model Comparison

```python
results = await testbench.compare_models(
    models=[
        "meta-llama/llama-3.2-3b-instruct",
        "deepseek/deepseek-r1",
        "qwen/qwen-2.5-72b-instruct"
    ],
    num_examples=50
)
```

### 3. Strategy Comparison

```python
results = await testbench.compare_strategies(
    strategies=["zero_shot", "few_shot", "chain_of_thought"],
    model="meta-llama/llama-3.2-3b-instruct",
    num_examples=50
)
```

## Command Line Usage

### Available Commands

```bash
# Quick test with custom parameters
python run_testbench.py quick --model deepseek/deepseek-r1 --strategy few_shot --examples 20

# List available models and presets
python run_testbench.py list-models
python run_testbench.py list-presets

# Run preset configurations
python run_testbench.py preset quick
python run_testbench.py preset model_comparison
python run_testbench.py preset comprehensive

# Create custom configuration
python run_testbench.py create-config --type default --name my_experiment
```

### Configuration Presets

- **`quick`**: 2 models, 2 strategies, 20 examples (~5 minutes)
- **`default`**: 3 models, 3 strategies, 100 examples (~30 minutes)
- **`model_comparison`**: 3 models, 1 strategy, 200 examples (~45 minutes)
- **`comprehensive`**: 6 models, 5 strategies, full dataset (~4+ hours)

## Understanding Results

### Output Locations

- **CSV Data**: `results/experiments.csv` - Detailed per-query results
- **Summary**: `results/summaries.csv` - Experiment-level metrics
- **Visualizations**: `plots/` - Performance charts and comparisons
- **Reports**: `results/*_report.txt` - Text summaries

### Key Metrics

- **Execution Accuracy**: Do predicted and gold SQL produce same results?
- **Exact Match**: Are SQL strings identical?
- **Component Match**: Are SQL components (SELECT, WHERE, etc.) correct?
- **Response Time**: How long did the model take?
- **Token Usage**: Cost tracking information

### Reading the Charts

The testbench generates several types of visualizations:

1. **Model Comparison**: Bar charts showing accuracy by model
2. **Strategy Comparison**: Effectiveness of different prompting approaches
3. **Database Performance**: Difficulty analysis across different databases
4. **Accuracy Heatmap**: Model vs Strategy performance matrix
5. **Performance Trends**: Metrics over time
6. **Token Efficiency**: Cost vs accuracy analysis

## Customization

### Custom Experiment Configuration

Create a YAML file in `configs/`:

```yaml
name: "my_experiment"
models:
  - name: "meta-llama/llama-3.2-3b-instruct"
    temperature: 0.1
    max_tokens: 1024
prompts:
  - strategy: "zero_shot"
  - strategy: "few_shot"
    num_examples: 5
dataset:
  max_examples: 100
  shuffle: true
```

Run with:
```bash
python run_testbench.py config my_experiment
```

### Available Models

Use `python run_testbench.py list-models` to see current options.

Popular choices:
- `meta-llama/llama-3.2-3b-instruct` (fast, cost-effective)
- `deepseek/deepseek-r1` (good reasoning)
- `qwen/qwen-2.5-72b-instruct` (high accuracy)

### Prompting Strategies

- **`zero_shot`**: Schema + question only
- **`few_shot`**: Includes example question-SQL pairs
- **`chain_of_thought`**: Step-by-step reasoning
- **`multi_stage`**: Analysis → Generation → Verification

## Troubleshooting

### Common Issues

**API Key Problems:**
```bash
# Check if API key is set correctly
cat secrets.txt
# Should show: NOVITA_API_KEY=your_key_here
```

**Missing Dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt
```

**Spider Dataset Issues:**
```bash
# Re-clone Spider dataset
rm -rf spider
git clone https://github.com/taoyds/spider.git
```

**Import Errors:**
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Check if testbench imports work
python -c "from src import NL2SQLTestbench; print('✅ Import successful')"
```

### Performance Tips

1. **Start Small**: Use `quick` preset first to validate setup
2. **Monitor Costs**: Check token usage in results before running large experiments
3. **Use Concurrency**: Adjust `max_concurrent_requests` in configs for faster processing
4. **Filter Data**: Use `specific_databases` or `max_examples` to focus testing
