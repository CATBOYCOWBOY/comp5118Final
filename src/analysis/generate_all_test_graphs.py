#!/usr/bin/env python3

import json
import os
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def load_results_data(results_dir):
    """Load all results.json files from the results directory."""
    results_data = {}

    pattern = os.path.join(results_dir, "*/results.json")
    results_files = glob.glob(pattern)

    for results_file in results_files:
        with open(results_file, 'r') as f:
            data = json.load(f)

        experiments = data['experiments']['experiments']
        for experiment_name, experiment_data in experiments.items():
            model_name = experiment_data.get('config', {}).get('model')
            if not model_name:
                model_name = experiment_name.split('_spider1_basic')[0]

            display_name = clean_model_name(model_name)

            summary = experiment_data['summary']
            results_data[display_name] = {
                'total_examples': summary['total_examples'],
                'successful_predictions': summary['successful_predictions'],
                'test_suite_accuracy': summary['test_suite_accuracy'],
                'exact_match_accuracy': summary['exact_match_accuracy']
            }

    return results_data

def clean_model_name(model_name):
    name = model_name.replace('meta-llama/', '').replace('qwen/', '').replace('google/', '').replace('deepseek/', '').replace('openai/', '')

    name = name.replace('_', ' ').replace('-', ' ')

    name_mappings = {
        'llama 3.1 8b instruct': 'LLaMA 3.1 8B instruct',
        'llama 3.2 3b instruct': 'LLaMA 3.2 3B instruct',
        'qwen3 4b fp8': 'Qwen3 4B fp8',
        'qwen3 8b fp8': 'Qwen3 8B fp8',
        'qwen3 32b fp8': 'Qwen3 32B fp8',
        'qwen2.5 7b instruct': 'Qwen2.5 7B instruct',
        'qwen3 coder 30b a3b instruct': 'Qwen3 Coder 30B instruct',
        'gemma 3 12b it': 'Gemma 3 12B it',
        'gemma 3 27b it': 'Gemma 3 27B it',
        'deepseek r1 distill qwen 14b': 'DeepSeek R1 14B',
        'gpt oss 20b': 'GPT OSS 20B'
    }

    return name_mappings.get(name.lower(), name.title())

def create_bar_chart(data, metric_key, title, ylabel, filename, color='steelblue'):
    models = list(data.keys())
    values = []

    if metric_key == 'successful_predictions_rate':
        values = [data[model]['successful_predictions'] / data[model]['total_examples']
                 for model in models]
    else:
        values = [data[model][metric_key] for model in models]

    sorted_pairs = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
    models, values = zip(*sorted_pairs)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, values, color=color, alpha=0.7, edgecolor='black', linewidth=0.8)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        if metric_key == 'successful_predictions_rate':
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.1%}', ha='center', va='bottom', fontsize=10)
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')

    if metric_key == 'successful_predictions_rate':
        plt.ylim(0, 1.1)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    else:
        plt.ylim(0, max(values) * 1.1)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

def main():
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "results" / "Batch 1"
    output_dir = project_root / "plots"

    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {results_dir}")
    print(f"Output directory: {output_dir}")

    results_data = load_results_data(str(results_dir))

    if not results_data:
        print("No results data found! Check the results directory path.")
        return

    print(f"Found data for {len(results_data)} models:")
    for model in sorted(results_data.keys()):
        data = results_data[model]
        print(f"  {model}: {data['successful_predictions']}/{data['total_examples']} successful, "
              f"Test Suite: {data['test_suite_accuracy']:.3f}, "
              f"Exact Match: {data['exact_match_accuracy']:.3f}")

    create_bar_chart(
        results_data,
        'successful_predictions_rate',
        'Model Performance: Successful Prediction Rate\n(Successfully Generated SQL / Total Examples)',
        'Successful Prediction Rate',
        str(output_dir / 'successful_predictions_rate.png'),
        color='darkgreen'
    )

    create_bar_chart(
        results_data,
        'test_suite_accuracy',
        'Model Performance: Test Suite Accuracy\n(Semantic Correctness via Execution on Multiple Databases)',
        'Test Suite Accuracy',
        str(output_dir / 'test_suite_accuracy.png'),
        color='steelblue'
    )

    create_bar_chart(
        results_data,
        'exact_match_accuracy',
        'Model Performance: Exact Match Accuracy\n(Syntactic/Structural Query Matching)',
        'Exact Match Accuracy',
        str(output_dir / 'exact_match_accuracy.png'),
        color='darkred'
    )

    print("\nAll graphs generated successfully!")
    print(f"Check the {output_dir} directory for the generated plots.")

if __name__ == "__main__":
    main()