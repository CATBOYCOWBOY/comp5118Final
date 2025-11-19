"""
Spider2-lite testbench for text-in-results-out NL2SQL evaluation.
"""
import asyncio
import os
import time
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(__file__))

from spider2 import Spider2Manager, Spider2Example
from strategies import Spider2Strategy
from novita_client import NovitaClient
from performance import PerformanceManager


class Spider2Testbench:
    """Simplified Spider2-lite testbench with text-in-results-out evaluation."""

    def __init__(self, spider2_path: str = "Spider2"):
        self.spider2_manager = Spider2Manager(spider2_path)
        self.performance_tracker = PerformanceManager()

        # Initialize Novita client
        self.novita_client = NovitaClient()
        print(f"Spider2 Testbench initialized with {len(self.spider2_manager.examples)} examples")

    async def run_experiment(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a Spider2-lite experiment with given configuration."""

        experiment_id = str(int(time.time() * 1000))  # Simple timestamp ID
        print(f"Started experiment: {config['name']}_{experiment_id}")

        # Get examples
        max_examples = config.get('max_examples')
        examples = self.spider2_manager.get_examples(max_examples)
        print(f"Loaded {len(examples)} examples")

        results = {}

        # Handle single strategy vs multiple strategies
        if 'strategy' in config:
            strategies = [config['strategy']]
        else:
            strategies = config.get('strategies', ['spider2_basic'])

        for model_name in config['models']:
            for strategy_name in strategies:
                print(f"Model: {model_name}, Strategy: {strategy_name}")

                # Create strategy instance
                strategy = Spider2Strategy(self.spider2_manager)

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
        strategy: Spider2Strategy,
        strategy_name: str,
        examples: List[Spider2Example],
        experiment_id: str
    ) -> Dict[str, Any]:
        """Run evaluation on examples with a specific model and strategy."""

        experiment_results = {
            "summary": {
                "total_examples": len(examples),
                "successful_executions": 0,
                "execution_accuracy": 0.0,
                "successful_examples": 0
            },
            "entries": []
        }

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

                # Evaluate execution
                result = strategy.evaluate_result(predicted_sql, example)

                # Create result entry
                entry = {
                    "instance_id": example.instance_id,
                    "question": example.question,
                    "database": example.db,
                    "predicted_sql": predicted_sql or "",
                    "extracted_sql": predicted_sql or "",
                    "execution_success": result.execution_result is not None,
                    "execution_match": result.execution_match,
                    "error": result.error,
                    "llm_response": llm_response,
                    "performance": {
                        "response_time": response_time,
                        "tokens_used": self._estimate_tokens(llm_response),
                        "model": model_name
                    }
                }

                if result.execution_result is not None:
                    successful_count += 1
                    entry["execution_rows"] = len(result.execution_result)
                    entry["execution_columns"] = len(result.execution_result.columns) if not result.execution_result.empty else 0

                experiment_results["entries"].append(entry)

                # Track performance (basic tracking)
                # self.performance_tracker would be used here for detailed analytics

            except Exception as e:
                print(f"Error processing example {i}: {e}")
                # Add error entry
                experiment_results["entries"].append({
                    "instance_id": example.instance_id,
                    "question": example.question,
                    "database": example.db,
                    "predicted_sql": "",
                    "extracted_sql": "",
                    "error": str(e),
                    "execution_success": False,
                    "execution_match": False,
                    "llm_response": "",
                    "performance": {
                        "response_time": 0.0,
                        "tokens_used": 0,
                        "model": model_name
                    }
                })

        # Update summary
        experiment_results["summary"]["successful_examples"] = successful_count
        experiment_results["summary"]["successful_executions"] = successful_count
        experiment_results["summary"]["execution_accuracy"] = successful_count / len(examples) if examples else 0.0

        return experiment_results

    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation (approximate)."""
        return len(text.split()) * 1.3  # Rough approximation

    async def quick_test(self, model_name: str, max_examples: int = 10) -> Dict[str, Any]:
        """Run a quick test with Spider2-lite."""
        config = {
            "name": "spider2_quick_test",
            "models": [model_name],
            "strategy": "spider2_basic",
            "max_examples": max_examples
        }

        print(f"Running Spider2 quick test: {model_name} with {max_examples} examples")
        results = await self.run_experiment(config)

        # Print summary
        for exp_name, exp_results in results.items():
            summary = exp_results["summary"]
            print(f"Results: {summary['execution_accuracy']:.3f} execution accuracy")
            print(f"Completed: {summary['successful_examples']}/{summary['total_examples']} examples")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/spider2_quick_test_{timestamp}_results.json"
        self.save_results({"config": config, "experiments": results}, output_file)

        return results

    async def compare_models(
        self,
        models: Optional[List[str]] = None,
        max_examples: int = 50
    ) -> Dict[str, Any]:
        """Compare multiple models on Spider2-lite."""

        if models is None:
            from .config import AVAILABLE_MODELS
            models = AVAILABLE_MODELS[:3]  # Limit to first 3 for speed

        config = {
            "name": "spider2_model_comparison",
            "models": models,
            "strategy": "spider2_basic",
            "max_examples": max_examples
        }

        print(f"Running Spider2 model comparison with {len(models)} models")
        results = await self.run_experiment(config)

        # Print comparison summary
        print("\nModel Comparison Results:")
        print("=" * 60)
        for exp_name, exp_results in results.items():
            summary = exp_results["summary"]
            print(f"{exp_name:30} | Accuracy: {summary['execution_accuracy']:.3f} | Success: {summary['successful_examples']}/{summary['total_examples']}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/spider2_model_comparison_{timestamp}_results.json"
        self.save_results({"config": config, "experiments": results}, output_file)

        return results

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the Spider2-lite dataset."""
        stats = self.spider2_manager.get_database_stats()

        print("Spider2-lite Dataset Information:")
        print("=" * 50)
        print(f"Total Examples: {stats['total_examples']}")
        print(f"Available Databases: {stats['available_databases']}")
        print(f"Examples with External Knowledge: {stats['examples_with_external_knowledge']}")
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
            "dataset_type": "spider2-lite",
            "experiments": results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {output_file}")