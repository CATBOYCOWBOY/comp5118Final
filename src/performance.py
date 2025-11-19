import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .novita_client import LLMResponse


class PerformanceManager:
    """Simplified performance tracking and visualization system."""

    def __init__(self, results_dir: str = "results", plots_dir: str = "plots"):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        self.results_file = os.path.join(results_dir, "results.json")
        self.current_experiment = None
        self.current_results = []

        # Set visualization style
        plt.style.use('default')
        sns.set_palette("husl")

    def start_experiment(self, model_name: str, strategy: str, experiment_id: Optional[str] = None) -> str:
        """Start a new experiment tracking session."""
        if not experiment_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{model_name}_{strategy}_{timestamp}"

        self.current_experiment = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat()
        }
        self.current_results = []

        print(f"Started experiment: {experiment_id}")
        return experiment_id

    def record_query_result(self, query_result, llm_response: LLMResponse, strategy: str, model_name: str) -> None:
        """Record a single query result."""
        if not self.current_experiment:
            raise ValueError("No experiment started. Call start_experiment() first.")

        usage = llm_response.usage or {}
        total_tokens = usage.get('total_tokens', 0)

        result = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "strategy": strategy,
            "accuracy": {
                "exact_match": query_result.exact_match,
                "execution_match": query_result.execution_match
            },
            "performance": {
                "response_time": llm_response.response_time,
                "tokens_used": total_tokens
            },
            "error": query_result.error or (llm_response.error if not llm_response.success else None)
        }

        self.current_results.append(result)

    def finish_experiment(self) -> Dict[str, Any]:
        """Finish the current experiment and save results."""
        if not self.current_experiment or not self.current_results:
            raise ValueError("No experiment data to save")

        # Calculate summary
        total = len(self.current_results)
        exact_matches = sum(1 for r in self.current_results if r["accuracy"]["exact_match"])
        execution_matches = sum(1 for r in self.current_results if r["accuracy"]["execution_match"])
        errors = sum(1 for r in self.current_results if r["error"])

        avg_response_time = sum(r["performance"]["response_time"] for r in self.current_results) / total
        total_tokens = sum(r["performance"]["tokens_used"] for r in self.current_results)

        summary = {
            **self.current_experiment,
            "results": {
                "total_queries": total,
                "exact_match_accuracy": exact_matches / total,
                "execution_accuracy": execution_matches / total,
                "error_rate": errors / total,
                "avg_response_time": avg_response_time,
                "total_tokens": total_tokens
            }
        }

        # Save to JSON
        self._save_results(summary)

        experiment_id = self.current_experiment["experiment_id"]
        print(f"Finished experiment: {experiment_id}")
        print(f"Results: {summary['results']['exact_match_accuracy']:.3f} exact match, "
              f"{summary['results']['execution_accuracy']:.3f} execution accuracy")

        self.current_experiment = None
        self.current_results = []

        return summary

    def _save_results(self, experiment_summary: Dict[str, Any]) -> None:
        """Save experiment results to JSON file."""
        results = []
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                results = json.load(f)

        results.append(experiment_summary)

        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def load_results(self) -> List[Dict[str, Any]]:
        """Load all experiment results from JSON."""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []

    def get_model_comparison(self) -> Dict[str, Dict[str, float]]:
        """Get comparison of model performance across experiments."""
        results = self.load_results()
        if not results:
            return {}

        model_stats = defaultdict(lambda: {
            "exact_match_accuracy": [],
            "execution_accuracy": [],
            "avg_response_time": [],
            "total_tokens": [],
            "error_rate": []
        })

        for experiment in results:
            model = experiment["model_name"]
            res = experiment["results"]
            model_stats[model]["exact_match_accuracy"].append(res["exact_match_accuracy"])
            model_stats[model]["execution_accuracy"].append(res["execution_accuracy"])
            model_stats[model]["avg_response_time"].append(res["avg_response_time"])
            model_stats[model]["total_tokens"].append(res["total_tokens"])
            model_stats[model]["error_rate"].append(res["error_rate"])

        # Calculate averages
        comparison = {}
        for model, stats in model_stats.items():
            comparison[model] = {
                "exact_match_accuracy_mean": sum(stats["exact_match_accuracy"]) / len(stats["exact_match_accuracy"]),
                "execution_accuracy_mean": sum(stats["execution_accuracy"]) / len(stats["execution_accuracy"]),
                "average_response_time_mean": sum(stats["avg_response_time"]) / len(stats["avg_response_time"]),
                "total_tokens_sum": sum(stats["total_tokens"]),
                "error_rate_mean": sum(stats["error_rate"]) / len(stats["error_rate"])
            }

        return comparison

    def get_strategy_comparison(self) -> Dict[str, Dict[str, float]]:
        """Get comparison of strategy performance across experiments."""
        results = self.load_results()
        if not results:
            return {}

        strategy_stats = defaultdict(lambda: {
            "exact_match_accuracy": [],
            "execution_accuracy": [],
            "avg_response_time": [],
            "total_tokens": [],
            "error_rate": []
        })

        for experiment in results:
            strategy = experiment["strategy"]
            res = experiment["results"]
            strategy_stats[strategy]["exact_match_accuracy"].append(res["exact_match_accuracy"])
            strategy_stats[strategy]["execution_accuracy"].append(res["execution_accuracy"])
            strategy_stats[strategy]["avg_response_time"].append(res["avg_response_time"])
            strategy_stats[strategy]["total_tokens"].append(res["total_tokens"])
            strategy_stats[strategy]["error_rate"].append(res["error_rate"])

        # Calculate averages
        comparison = {}
        for strategy, stats in strategy_stats.items():
            comparison[strategy] = {
                "exact_match_accuracy_mean": sum(stats["exact_match_accuracy"]) / len(stats["exact_match_accuracy"]),
                "execution_accuracy_mean": sum(stats["execution_accuracy"]) / len(stats["execution_accuracy"]),
                "average_response_time_mean": sum(stats["avg_response_time"]) / len(stats["avg_response_time"]),
                "total_tokens_sum": sum(stats["total_tokens"]),
                "error_rate_mean": sum(stats["error_rate"]) / len(stats["error_rate"])
            }

        return comparison

    def plot_model_comparison(self, save: bool = True, show: bool = False) -> str:
        """Create a simple model comparison plot."""
        model_comparison = self.get_model_comparison()

        if not model_comparison:
            print("No data available for model comparison")
            return ""

        models = list(model_comparison.keys())
        exact_match = [model_comparison[m]['exact_match_accuracy_mean'] for m in models]
        execution_match = [model_comparison[m]['execution_accuracy_mean'] for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy comparison
        x = np.arange(len(models))
        width = 0.35

        ax1.bar(x - width/2, exact_match, width, label='Exact Match', alpha=0.8)
        ax1.bar(x + width/2, execution_match, width, label='Execution Match', alpha=0.8)

        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Response time comparison
        response_times = [model_comparison[m]['average_response_time_mean'] for m in models]
        ax2.bar(models, response_times, alpha=0.8, color='orange')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Response Time (seconds)')
        ax2.set_title('Average Response Time by Model')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = os.path.join(self.plots_dir, 'model_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to: {filename}")

        if show:
            plt.show()
        else:
            plt.close()

        return filename if save else ""

    def plot_strategy_comparison(self, save: bool = True, show: bool = False) -> str:
        """Create a strategy comparison plot."""
        strategy_comparison = self.get_strategy_comparison()
        if not strategy_comparison:
            print("No data available for strategy comparison")
            return ""

        strategies = list(strategy_comparison.keys())
        accuracies = [strategy_comparison[s]["exact_match_accuracy_mean"] for s in strategies]
        response_times = [strategy_comparison[s]["average_response_time_mean"] for s in strategies]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy plot
        bars1 = ax1.bar(strategies, accuracies, color='lightcoral', alpha=0.7)
        ax1.set_ylabel('Exact Match Accuracy')
        ax1.set_title('Accuracy by Strategy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(accuracies) * 1.1 if accuracies else 1)

        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # Response time plot
        bars2 = ax2.bar(strategies, response_times, color='lightblue', alpha=0.7)
        ax2.set_ylabel('Response Time (seconds)')
        ax2.set_title('Average Response Time by Strategy')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, time in zip(bars2, response_times):
            height = bar.get_height()
            ax2.annotate(f'{time:.2f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.tight_layout()

        filename = ""
        if save:
            filename = os.path.join(self.plots_dir, 'strategy_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Strategy comparison plot saved to: {filename}")

        if show:
            plt.show()
        else:
            plt.close()

        return filename if save else ""

    def plot_accuracy_comparison(self, save: bool = True, show: bool = False) -> str:
        """Create accuracy comparison plot (both model and strategy)."""
        model_comparison = self.get_model_comparison()
        strategy_comparison = self.get_strategy_comparison()

        if not model_comparison and not strategy_comparison:
            print("No data available for accuracy comparison")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Model accuracy comparison
        if model_comparison:
            models = list(model_comparison.keys())
            model_accuracies = [model_comparison[m]["exact_match_accuracy_mean"] for m in models]

            bars1 = axes[0].bar(models, model_accuracies, color='lightgreen', alpha=0.7)
            axes[0].set_ylabel('Exact Match Accuracy')
            axes[0].set_title('Accuracy by Model')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, max(model_accuracies) * 1.1 if model_accuracies else 1)

            # Add value labels
            for bar, acc in zip(bars1, model_accuracies):
                height = bar.get_height()
                axes[0].annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # Strategy accuracy comparison
        if strategy_comparison:
            strategies = list(strategy_comparison.keys())
            strategy_accuracies = [strategy_comparison[s]["exact_match_accuracy_mean"] for s in strategies]

            bars2 = axes[1].bar(strategies, strategy_accuracies, color='lightsalmon', alpha=0.7)
            axes[1].set_ylabel('Exact Match Accuracy')
            axes[1].set_title('Accuracy by Strategy')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, max(strategy_accuracies) * 1.1 if strategy_accuracies else 1)

            # Add value labels
            for bar, acc in zip(bars2, strategy_accuracies):
                height = bar.get_height()
                axes[1].annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.tight_layout()

        filename = ""
        if save:
            filename = os.path.join(self.plots_dir, 'accuracy_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Accuracy comparison plot saved to: {filename}")

        if show:
            plt.show()
        else:
            plt.close()

        return filename if save else ""

    def plot_performance_comparison(self, save: bool = True, show: bool = False) -> str:
        """Create performance comparison plot (response time and token usage)."""
        model_comparison = self.get_model_comparison()
        strategy_comparison = self.get_strategy_comparison()

        if not model_comparison and not strategy_comparison:
            print("No data available for performance comparison")
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Model response times
        if model_comparison:
            models = list(model_comparison.keys())
            model_times = [model_comparison[m]["average_response_time_mean"] for m in models]
            model_tokens = [model_comparison[m]["total_tokens_sum"] for m in models]

            axes[0,0].bar(models, model_times, color='skyblue', alpha=0.7)
            axes[0,0].set_ylabel('Response Time (seconds)')
            axes[0,0].set_title('Response Time by Model')
            axes[0,0].tick_params(axis='x', rotation=45)
            axes[0,0].grid(True, alpha=0.3)

            axes[0,1].bar(models, model_tokens, color='lightcoral', alpha=0.7)
            axes[0,1].set_ylabel('Total Tokens Used')
            axes[0,1].set_title('Token Usage by Model')
            axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].grid(True, alpha=0.3)

        # Strategy performance
        if strategy_comparison:
            strategies = list(strategy_comparison.keys())
            strategy_times = [strategy_comparison[s]["average_response_time_mean"] for s in strategies]
            strategy_tokens = [strategy_comparison[s]["total_tokens_sum"] for s in strategies]

            axes[1,0].bar(strategies, strategy_times, color='lightgreen', alpha=0.7)
            axes[1,0].set_ylabel('Response Time (seconds)')
            axes[1,0].set_title('Response Time by Strategy')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)

            axes[1,1].bar(strategies, strategy_tokens, color='lightsalmon', alpha=0.7)
            axes[1,1].set_ylabel('Total Tokens Used')
            axes[1,1].set_title('Token Usage by Strategy')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = ""
        if save:
            filename = os.path.join(self.plots_dir, 'performance_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to: {filename}")

        if show:
            plt.show()
        else:
            plt.close()

        return filename if save else ""

    def create_dashboard(self, save: bool = True) -> List[str]:
        """Create separate visualization files."""
        print("Generating performance visualizations...")

        plots = []
        plots.append(self.plot_model_comparison(save, False))
        plots.append(self.plot_strategy_comparison(save, False))
        plots.append(self.plot_accuracy_comparison(save, False))
        plots.append(self.plot_performance_comparison(save, False))

        plots = [p for p in plots if p]
        print(f"Visualization complete. {len(plots)} separate PNG files generated in {self.plots_dir}/")
        return plots

    def generate_summary_report(self) -> str:
        """Generate a text summary report of performance."""
        results = self.load_results()

        if not results:
            return "No experiment data available."

        total_experiments = len(results)
        avg_exact_match = sum(r["results"]["exact_match_accuracy"] for r in results) / total_experiments
        avg_execution = sum(r["results"]["execution_accuracy"] for r in results) / total_experiments
        total_tokens = sum(r["results"]["total_tokens"] for r in results)

        report = f"""
NL2SQL Testbench Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
========
Total Experiments: {total_experiments}
Average Exact Match Accuracy: {avg_exact_match:.3f}
Average Execution Accuracy: {avg_execution:.3f}
Total Tokens Used: {total_tokens:,}

EXPERIMENTS
===========
"""

        for result in results:
            report += f"- {result['experiment_id']}: {result['results']['execution_accuracy']:.3f} accuracy\n"

        return report

    def export_results(self) -> str:
        """Export all results as JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = os.path.join(self.results_dir, f"export_{timestamp}.json")

        results = self.load_results()
        with open(export_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results exported to: {export_file}")
        return export_file