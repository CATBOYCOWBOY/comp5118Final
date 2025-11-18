# NL2SQL Testbench

A comprehensive framework for evaluating Large Language Models (3-20B parameters) on Natural Language to SQL translation using the Spider dataset and Novita AI's API service.

## ⚡ Quick Start

```bash
# Automated setup
./startup.sh

# Add your API key
echo "NOVITA_API_KEY=your_api_key_here" > secrets.txt

# Run your first test
source .venv/bin/activate
python run_testbench.py quick
```

**📖 New here?** See [getting_started.md](getting_started.md) for detailed setup instructions and examples.

## 🎯 Features

- **Multi-Model Evaluation**: Llama, Deepseek, Qwen models via Novita AI
- **Advanced Prompting**: Zero-shot, few-shot, chain-of-thought, and multi-stage strategies
- **Comprehensive Metrics**: Execution accuracy, exact match, component analysis
- **Rich Visualizations**: Performance charts, model comparisons, trend analysis
- **Flexible Configuration**: YAML/JSON experiment configs with preset options
- **Production Ready**: Async processing, progress tracking, cost monitoring

## 🏃 Quick Examples

```bash
# Quick test (5 examples, ~2 minutes)
python run_testbench.py quick

# Compare 3 models (50 examples, ~15 minutes)
python run_testbench.py preset model_comparison

# Compare prompting strategies
python run_testbench.py preset strategy_comparison
```

## 🤖 Supported Models

| Family | Models | Best For |
|--------|--------|----------|
| **Llama** | 3.2-3B, 3.2-11B, 3.2-90B | General purpose, cost-effective |
| **Deepseek** | R1, V3 | Code reasoning, complex queries |
| **Qwen** | 2.5-72B, 2.5-Coder-32B | High accuracy, specialized tasks |

## 📊 Evaluation Capabilities

- **Execution Accuracy**: Tests if SQL produces correct results
- **Component Analysis**: Evaluates SQL structure (SELECT, WHERE, etc.)
- **Performance Metrics**: Response time, token usage, cost tracking
- **Error Analysis**: Categorizes and analyzes failure patterns
- **Database Difficulty**: Performance across simple → complex schemas

## 📈 Output & Results

Results are automatically saved in multiple formats:

- **CSV/Excel**: Detailed per-query and summary data
- **Visualizations**: Model comparisons, strategy analysis, trend charts
- **Reports**: Text summaries with key insights
- **JSON**: Programmatic access to all metrics

## ⚙️ Configuration

Create custom experiments with YAML configs:

```yaml
name: "my_experiment"
models:
  - name: "meta-llama/llama-3.2-3b-instruct"
  - name: "deepseek/deepseek-r1"
prompts:
  - strategy: "zero_shot"
  - strategy: "few_shot"
    num_examples: 3
dataset:
  max_examples: 100
  shuffle: true
```

Available presets: `quick`, `default`, `model_comparison`, `comprehensive`

## 🚀 Command Line Interface

```bash
# Basic usage
python run_testbench.py quick --examples 20
python run_testbench.py preset comprehensive

# Utilities
python run_testbench.py list-models
python run_testbench.py create-config --type default --name my_test

# Custom configs
python run_testbench.py config my_experiment
```

## 📁 Project Structure

```
├── src/                    # Core implementation (flat structure)
│   ├── testbench.py       # Main orchestrator
│   ├── novita_client.py   # Novita AI integration
│   ├── strategies.py      # Prompting strategies
│   ├── spider_loader.py   # Dataset handling
│   └── ...               # Other core modules
├── examples/              # Usage examples and tutorials
├── spider/               # Spider dataset (auto-downloaded)
├── results/              # Experiment outputs
├── plots/                # Generated visualizations
├── configs/              # Configuration files
└── getting_started.md    # Detailed setup guide
```

## 🔧 Development & Extension

The testbench is designed for easy extension:

- **New Models**: Add to `ModelRegistry` in `novita_client.py`
- **Prompting Strategies**: Implement `PromptStrategy` interface
- **Custom Metrics**: Extend `SpiderEvaluator` class
- **Visualizations**: Add plots to `PerformanceVisualizer`

**📚 Detailed setup and examples:** See [getting_started.md](getting_started.md) for step-by-step instructions.

## 💰 Cost Management

- Built-in token usage tracking
- Cost estimation for each model
- Configurable concurrency limits
- Progress monitoring for large experiments

## 📄 Citation

Built on the Spider dataset:

```bibtex
@inproceedings{Yu&al.18c,
  title={Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author={Tao Yu and Rui Zhang and Kai Yang and others},
  booktitle="EMNLP", year=2018
}
```

---

**🚀 Ready to start?** Run `./startup.sh` and follow the [getting started guide](getting_started.md)!
