#!/usr/bin/env python3

import json
import os
import matplotlib.pyplot as plt
import glob
import statistics
from pathlib import Path

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

def load_token_usage_data(results_dir):
    token_data = {}
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
            entries = experiment_data.get('entries', [])
            token_usages = []

            for entry in entries:
                performance = entry.get('performance', {})
                tokens_used = performance.get('tokens_used')
                if tokens_used is not None and tokens_used > 0:
                    token_usages.append(tokens_used)

            if token_usages:
                token_data[display_name] = {
                    'mean_tokens': statistics.mean(token_usages)
                }

    return token_data

def create_token_usage_bar_chart(data, output_path):
    models = list(data.keys())
    mean_tokens = [data[model]['mean_tokens'] for model in models]
    sorted_pairs = sorted(zip(models, mean_tokens), key=lambda x: x[1])
    models, mean_tokens = zip(*sorted_pairs)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, mean_tokens, color='darkorange', alpha=0.7, edgecolor='black', linewidth=0.8)

    for bar, value in zip(bars, mean_tokens):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)

    plt.title('Model Token Efficiency: Mean Tokens Used per Question\n(Lower is More Efficient)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Tokens per Question', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(mean_tokens) * 1.1)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")


project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "Batch 1"
output_dir = project_root / "plots"

output_dir.mkdir(exist_ok=True)


token_data = load_token_usage_data(str(results_dir))

if token_data:
    output_path = str(output_dir / 'mean_token_usage.png')
    create_token_usage_bar_chart(token_data, output_path)