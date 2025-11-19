"""
Spider1 testbench for traditional Text2SQL evaluation with test-suite-sql-eval.
"""
import asyncio
import os
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(__file__))

from spider1 import Spider1Manager, Spider1Example
from prompting_strategy import Spider1Strategy
from novita_client import NovitaClient
from performance import PerformanceManager


class Spider1Testbench:
    """Spider1 testbench with test-suite-sql-eval evaluation."""

    def __init__(self, spider_path: str = "spider", test_suite_path: str = "test-suite-sql-eval"):
        self.spider1_manager = Spider1Manager(spider_path, test_suite_path)
        self.performance_tracker = PerformanceManager()

        # Initialize Novita client
        self.novita_client = NovitaClient()
        print(f"Spider1 Testbench initialized with {len(self.spider1_manager.examples)} examples")

    async def run_experiment(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a Spider1 experiment with given configuration."""

        experiment_id = str(int(time.time() * 1000))  # Simple timestamp ID
        print(f"Started experiment: {config['name']}_{experiment_id}")

        # Get examples
        max_examples = config.get('max_examples')
        examples = self.spider1_manager.get_examples(max_examples)
        print(f"Loaded {len(examples)} examples")

        results = {}

        # Handle single strategy vs multiple strategies
        if 'strategy' in config:
            strategies = [config['strategy']]
        else:
            strategies = config.get('strategies', ['spider1_basic'])

        for model_name in config['models']:
            for strategy_name in strategies:
                print(f"Model: {model_name}, Strategy: {strategy_name}")

                # Create strategy instance
                strategy = Spider1Strategy(self.spider1_manager)

                # Run evaluation
                experiment_result = await self._run_single_experiment(
                    model_name, strategy, strategy_name, examples, experiment_id
                )

                results[f"{model_name.split('/')[-1]}_{strategy_name}"] = experiment_result

        print(f"Finished experiment: {config['name']}_{experiment_id}")
        return results

    async def _run_single_experiment(
        self,
        model_name: str,
        strategy: 'Spider1Strategy',
        strategy_name: str,
        examples: List[Spider1Example],
        experiment_id: str
    ) -> Dict[str, Any]:
        """Run evaluation on examples with a specific model and strategy."""

        experiment_results = {
            "summary": {
                "total_examples": len(examples),
                "successful_predictions": 0,
                "test_suite_accuracy": 0.0,
                "exact_match_accuracy": 0.0
            },
            "entries": []
        }

        predictions = []
        gold_queries = []
        successful_count = 0

        for i, example in enumerate(examples):
            try:
                print(f"Processing example {i+1}/{len(examples)}: {example.instance_id}")

                # Generate prompt
                prompt = strategy.generate_prompt(example)

                # Get LLM response
                start_time = time.time()
                response = await self.novita_client.complete_async(
                    prompt=prompt,
                    model=model_name,
                    temperature=0.1,
                    max_tokens=1024
                )
                llm_response = response.content
                response_time = time.time() - start_time

                # Extract SQL
                predicted_sql = strategy.process_prompt_to_sql(llm_response)

                if predicted_sql:
                    predictions.append((predicted_sql, example.db_id))
                    gold_queries.append((example.query, example.db_id))
                    successful_count += 1
                else:
                    predictions.append(("SELECT 1", example.db_id))  # Dummy query
                    gold_queries.append((example.query, example.db_id))

                # Create result entry
                entry = {
                    "instance_id": example.instance_id,
                    "question": example.question,
                    "database": example.db_id,
                    "predicted_sql": predicted_sql or "",
                    "gold_sql": example.query,
                    "llm_response": llm_response,
                    "performance": {
                        "response_time": response_time,
                        "tokens_used": self._estimate_tokens(llm_response),
                        "model": model_name
                    }
                }

                experiment_results["entries"].append(entry)

            except Exception as e:
                print(f"Error processing example {i}: {e}")
                # Add error entry
                experiment_results["entries"].append({
                    "instance_id": example.instance_id,
                    "question": example.question,
                    "database": example.db_id,
                    "predicted_sql": "",
                    "gold_sql": example.query,
                    "error": str(e),
                    "llm_response": "",
                    "performance": {
                        "response_time": 0.0,
                        "tokens_used": 0,
                        "model": model_name
                    }
                })

                # Still add to evaluation lists with dummy query
                predictions.append(("SELECT 1", example.db_id))
                gold_queries.append((example.query, example.db_id))

        # Run test-suite evaluation
        print("Running test-suite evaluation...")
        eval_results = self.spider1_manager.evaluate_with_test_suite(predictions, gold_queries)

        # Update summary with test-suite results
        experiment_results["summary"]["successful_predictions"] = successful_count
        experiment_results["summary"]["test_suite_accuracy"] = eval_results.get("test_suite_accuracy", 0.0)
        experiment_results["summary"]["exact_match_accuracy"] = eval_results.get("exact_match_accuracy", 0.0)
        experiment_results["evaluation_details"] = eval_results

        return experiment_results

    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation (approximate)."""
        return len(text.split()) * 1.3  # Rough approximation

    async def quick_test(self, model_name: str, max_examples: int = 10) -> Dict[str, Any]:
        """Run a quick test with Spider1."""
        config = {
            "name": "spider1_quick_test",
            "models": [model_name],
            "strategy": "spider1_basic",
            "max_examples": max_examples
        }

        print(f"Running Spider1 quick test: {model_name} with {max_examples} examples")
        results = await self.run_experiment(config)

        # Print summary
        for exp_name, exp_results in results.items():
            summary = exp_results["summary"]
            print(f"Results:")
            print(f"  Test-suite accuracy: {summary['test_suite_accuracy']:.3f}")
            print(f"  Exact-match accuracy: {summary['exact_match_accuracy']:.3f}")
            print(f"  Successful predictions: {summary['successful_predictions']}/{summary['total_examples']}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/spider1_quick_test_{timestamp}_results.json"
        self.save_results({"config": config, "experiments": results}, output_file)

        return results

    async def compare_models(
        self,
        models: Optional[List[str]] = None,
        max_examples: int = 50
    ) -> Dict[str, Any]:
        """Compare multiple models on Spider1."""

        if models is None:
            from .config import AVAILABLE_MODELS
            models = AVAILABLE_MODELS[:3]  # Limit to first 3 for speed

        config = {
            "name": "spider1_model_comparison",
            "models": models,
            "strategy": "spider1_basic",
            "max_examples": max_examples
        }

        print(f"Running Spider1 model comparison with {len(models)} models")
        results = await self.run_experiment(config)

        # Print comparison summary
        print("\nModel Comparison Results:")
        print("=" * 80)
        for exp_name, exp_results in results.items():
            summary = exp_results["summary"]
            print(f"{exp_name:30} | Test-suite: {summary['test_suite_accuracy']:.3f} | Exact: {summary['exact_match_accuracy']:.3f} | Success: {summary['successful_predictions']}/{summary['total_examples']}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/spider1_model_comparison_{timestamp}_results.json"
        self.save_results({"config": config, "experiments": results}, output_file)

        return results

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the Spider1 dataset."""
        stats = self.spider1_manager.get_database_stats()

        print("Spider1 Dataset Information:")
        print("=" * 50)
        print(f"Total Examples: {stats['total_examples']}")
        print(f"Total Databases: {stats['total_databases']}")
        print("\nDatabase Distribution:")
        for db, count in sorted(stats['database_distribution'].items()):
            print(f"  {db}: {count} examples")

        return stats

    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save experiment results to file."""
        import json

        # Ensure results directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Add metadata
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "dataset_type": "spider1",
            "experiments": results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {output_file}")