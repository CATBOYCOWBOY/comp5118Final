#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

model_sizes = [3, 4, 7, 8, 8, 14.5, 21, 27]  # in billions of parameters
execution_accuracy = [0.537, 0.506, 0.569, 0.632, 0.464, 0.538, 0.522, 0.570]  # execution accuracy rates
model_names = [
    "Llama 3.2 3B",
    "Qwen3 4B",
    "Qwen2.5 7B",
    "Llama 3.1 8B",
    "Qwen3 8B",
    "DeepSeek R1 14B",
    "GPT-OSS 20B",
    "Gemma3 27B"
]

plt.figure(figsize=(12, 8))

plt.scatter(model_sizes, execution_accuracy, s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1)

for size, acc, name in zip(model_sizes, execution_accuracy, model_names):
    plt.annotate(name,
                (size, acc),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.xlabel('Model Size (Billion Parameters)', fontsize=12, fontweight='bold')
plt.ylabel('Execution Accuracy', fontsize=12, fontweight='bold')
plt.title('NL2SQL Execution Accuracy vs Model Size\nLightweight Open Source Models on SPIDER-dev',
          fontsize=14, fontweight='bold', pad=20)

plt.xlim(0, 30)
plt.ylim(0.4, 0.65)

plt.grid(True, alpha=0.3, linestyle='--')

z = np.polyfit(model_sizes, execution_accuracy, 1)
p = np.poly1d(z)
plt.plot(model_sizes, p(model_sizes), "r--", alpha=0.5, linewidth=1,
         label=f'Trend line (R² = {np.corrcoef(model_sizes, execution_accuracy)[0,1]**2:.3f})')

plt.legend(loc='upper right', fontsize=10)

plt.tight_layout()

output_dir = Path(__file__).parent.parent.parent / "plots"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "execution_accuracy_vs_model_size.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

plt.show()