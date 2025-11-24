"""
Spider1 testbench for traditional Text2SQL evaluation with test-suite-sql-eval.
"""
import asyncio
import os
import time
import json
import logging
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
    MAX_STORED_RESPONSE_CHARS = 4000  # prevent huge result payloads

    def __init__(self, spider_path: str = "spider", test_suite_path: str = "test-suite-sql-eval"):
        self.spider1_manager = Spider1Manager(spider_path, test_suite_path)
        self.performance_tracker = PerformanceManager()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        self.logger = logging.getLogger("Spider1Testbench")

        # Initialize Novita client
        self.novita_client = NovitaClient()
        print(f"Spider1 Testbench initialized with {len(self.spider1_manager.examples)} examples")

    async def run_experiment(
        self,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """Run evaluation on examples for given model(s) and strategy(ies)."""

        experiment_id = str(int(time.time() * 1000))  # Simple timestamp ID
        run_dir = os.path.join("results", f"{config['name']}_{experiment_id}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Started experiment: {config['name']}_{experiment_id}")

        # Get examples
        max_examples = config.get('max_examples')
        examples = self.spider1_manager.get_examples(max_examples)
        print(f"Loaded {len(examples)} examples")

        results: Dict[str, Any] = {}

        strategies = [config['strategy']] if 'strategy' in config else config.get('strategies', ['spider1_basic'])

        for model_name in config['models']:
            for strategy_name in strategies:
                print(f"Model: {model_name}, Strategy: {strategy_name}")

                # Create strategy instance
                strategy = Spider1Strategy(self.spider1_manager)

                # Create results directory for this model/strategy combination
                results_dir = self.spider1_manager.create_results_directory(
                    run_dir, model_name, strategy_name
                )

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
                        stored_response = llm_response
                        if len(stored_response) > self.MAX_STORED_RESPONSE_CHARS:
                            stored_response = stored_response[:self.MAX_STORED_RESPONSE_CHARS]

                        entry = {
                            "instance_id": example.instance_id,
                            "question": example.question,
                            "database": example.db_id,
                            "predicted_sql": predicted_sql or "",
                            "gold_sql": example.query,
                            "llm_response": stored_response,
                            "llm_response_truncated": len(llm_response) > len(stored_response),
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

                # Run test-suite evaluation (see test-suite-sql-eval README for format)
                print("Running test-suite evaluation...")
                eval_options = {
                    "etype": "all",
                    "plug_value": True,
                    "keep_distinct": False,
                }
                if config.get("evaluation"):
                    eval_options.update(config["evaluation"])
                self.logger.info(
                    "Evaluating %d predictions with test-suite-sql-eval (etype=%s, plug_value=%s, keep_distinct=%s)",
                    len(predictions),
                    eval_options["etype"],
                    eval_options["plug_value"],
                    eval_options["keep_distinct"],
                )
                eval_results = self.spider1_manager.evaluate_with_test_suite(
                    predictions,
                    gold_queries,
                    results_dir,
                    plug_value=eval_options["plug_value"],
                    keep_distinct=eval_options["keep_distinct"],
                    etype=eval_options["etype"]
                )
                self.logger.info(
                    "Evaluation completed: exec=%.3f exact=%.3f",
                    eval_results.get("test_suite_accuracy", 0.0),
                    eval_results.get("exact_match_accuracy", 0.0),
                )

                # Save experiment metadata to results directory
                metadata_file = os.path.join(results_dir, "metadata.json")
                metadata = {
                    "model_name": model_name,
                    "strategy_name": strategy_name,
                    "experiment_id": experiment_id,
                    "total_examples": len(examples),
                    "successful_predictions": successful_count,
                    "timestamp": experiment_results["entries"][0]["performance"]["response_time"] if experiment_results["entries"] else None
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Update summary with test-suite results
                experiment_results["summary"]["successful_predictions"] = successful_count
                experiment_results["summary"]["test_suite_accuracy"] = eval_results.get("test_suite_accuracy", 0.0)
                experiment_results["summary"]["exact_match_accuracy"] = eval_results.get("exact_match_accuracy", 0.0)
                experiment_results["evaluation_details"] = eval_results

                results[f"{model_name.split('/')[-1]}_{strategy_name}"] = experiment_results

        print(f"Finished experiment: {config['name']}_{experiment_id}")
        return results, run_dir

    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation (approximate)."""
        return len(text.split()) * 1.3  # Rough approximation

    async def run_test(self, model_name: str, max_examples: int = 10) -> Dict[str, Any]:
        """Run a test with Spider1."""
        config = {
            "name": "spider1_run_test",
            "models": [model_name],
            "strategy": "spider1_basic",
            "max_examples": max_examples
        }

        print(f"Running Spider1 test: {model_name} with {max_examples} examples")
        results, run_dir = await self.run_experiment(config)

        # Print summary
        for exp_name, exp_results in results.items():
            summary = exp_results["summary"]
            print(f"Results:")
            print(f"  Test-suite accuracy: {summary['test_suite_accuracy']:.3f}")
            print(f"  Exact-match accuracy: {summary['exact_match_accuracy']:.3f}")
            print(f"  Successful predictions: {summary['successful_predictions']}/{summary['total_examples']}")

        # Save results inside the run directory
        output_file = os.path.join(run_dir, "results.json")
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
