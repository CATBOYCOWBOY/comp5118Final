import asyncio
import time
import os
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from tqdm import tqdm
import random

from .novita_client import NovitaClient, LLMResponse
from .spider import SpiderManager, SpiderExample, ExampleSelector, QueryResult
from .strategies import MultiStageStrategy
from .performance import PerformanceManager
from . import config


class NL2SQLTestbench:
    def __init__(
        self,
        spider_path: str = "spider",
        results_dir: str = "results",
        api_key: Optional[str] = None
    ):
        self.spider_path = spider_path
        self.results_dir = results_dir

        self.client = NovitaClient(api_key)
        self.spider_manager = SpiderManager(spider_path)
        self.performance_manager = PerformanceManager(results_dir)

        self.example_selector = ExampleSelector(self.spider_manager)
        if not api_key:
            try:
                self.client.load_api_key()
                print("API key loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load API key: {e}")

        print(f"Testbench initialized with Spider dataset: {spider_path}")

    async def run_experiment(
        self,
        experiment_config: dict,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Run a complete experiment based on configuration."""
        print(f"\nStarting experiment: {experiment_config['name']}")

        if not config.validate_config(experiment_config):
            raise ValueError("Invalid configuration")

        print("Loading dataset...")
        examples = self.spider_manager.load_split(experiment_config.get("split", "dev"))

        max_examples = experiment_config.get("max_examples")
        if max_examples and len(examples) > max_examples:
            examples = random.sample(examples, max_examples)

        print(f"Loaded {len(examples)} examples")

        models = experiment_config["models"]

        # Handle both single strategy and multiple strategies
        if "strategy" in experiment_config:
            strategies = [experiment_config["strategy"]]
        else:
            strategies = experiment_config.get("strategies", ["multi_stage"])

        total_experiments = len(models) * len(strategies)
        experiment_results = {}
        current_experiment = 0

        for model_name in models:
            for strategy_name in strategies:
                current_experiment += 1

                print(f"\nExperiment {current_experiment}/{total_experiments}")
                print(f"Model: {model_name}")
                print(f"Strategy: {strategy_name}")

                experiment_key = f"{model_name.split('/')[-1]}_{strategy_name}"
                self.performance_manager.start_experiment(
                    model_name,
                    strategy_name,
                    f"{experiment_config['name']}_{experiment_key}_{int(time.time())}"
                )

                try:
                    results = await self._run_single_experiment(
                        examples,
                        model_name,
                        strategy_name,
                        progress_callback
                    )

                    experiment_results[experiment_key] = results

                    # Finish tracking
                    summary = self.performance_manager.finish_experiment()
                    print(f"Completed: {summary['results']['exact_match_accuracy']:.3f} accuracy")

                except Exception as e:
                    print(f"Experiment failed: {e}")
                    experiment_results[experiment_key] = {"error": str(e)}

        # Generate results
        final_results = {
            "config": experiment_config,
            "experiments": experiment_results,
            "summary": self._generate_experiment_summary(),
            "timestamp": datetime.now().isoformat()
        }

        # Save results and generate visualizations
        await self._save_experiment_results(final_results, experiment_config["name"])
        print("\nGenerating visualizations...")
        self.performance_manager.create_dashboard(save=True)

        print(f"\nExperiment '{experiment_config['name']}' completed successfully!")
        return final_results

    async def _run_single_experiment(
        self,
        examples: List[SpiderExample],
        model_name: str,
        strategy_name: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Run a single model-strategy experiment."""

        # Get strategy configuration
        strategy_config = config.get_strategy_config(strategy_name)
        strategy = MultiStageStrategy(**strategy_config)

        # Get model parameters
        model_params = config.get_model_params(model_name)

        query_results = []
        llm_responses = []
        extracted_sqls = []

        # Use tqdm for progress tracking
        pbar = tqdm(examples, desc=f"{model_name.split('/')[-1]} + {strategy_name}")

        for i, example in enumerate(pbar):
            try:
                # Generate prompt
                schema = self.spider_manager.get_schema(example.db_id)
                if not schema:
                    print(f"Warning: No schema found for {example.db_id}")
                    continue

                # Run multi-stage prompting
                query_result, llm_response, extracted_sql = await self._handle_multistage_prompting(
                    example, schema, strategy, model_params
                )

                # Always append to maintain list synchronization
                if query_result and llm_response:
                    query_results.append(query_result)
                    llm_responses.append(llm_response)
                    extracted_sqls.append(extracted_sql if extracted_sql else None)

                    # Record in tracker
                    self.performance_manager.record_query_result(
                        query_result,
                        llm_response,
                        strategy_name,
                        model_name
                    )
                else:
                    # Debug logging for failed queries
                    print(f"\n=== DEBUG: Failed query {i+1} ===")
                    print(f"Model: {model_name}")
                    print(f"Question: {example.question}")
                    print(f"Database: {example.db_id}")
                    if llm_response:
                        print(f"LLM Response Success: {llm_response.success}")
                        print(f"LLM Response Error: {llm_response.error}")
                        print(f"LLM Response Content: {repr(llm_response.content[:500])}...")
                    else:
                        print("LLM Response: None")
                    if query_result:
                        print(f"Query Result Error: {query_result.error}")
                        print(f"Extracted SQL: {repr(query_result.predicted_sql)}")
                    else:
                        print("Query Result: None")
                    print("================================\n")


                # Update progress
                if progress_callback:
                    progress_callback(i + 1, len(examples))

                # Update progress bar with current accuracy
                if query_results:
                    current_accuracy = sum(1 for r in query_results if r.execution_match) / len(query_results)
                    pbar.set_postfix({"accuracy": f"{current_accuracy:.3f}"})

                # Rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue

        # Calculate summary metrics
        total_examples = len(query_results)
        exact_matches = sum(1 for r in query_results if r.exact_match)
        component_matches = sum(1 for r in query_results if r.component_match)
        execution_matches = sum(1 for r in query_results if r.execution_match)

        # Merge query results with LLM responses for cleaner output
        merged_entries = []
        for i, (query_result, llm_response, extracted_sql) in enumerate(zip(query_results, llm_responses, extracted_sqls)):
            merged_entry = {
                "entry_id": i + 1,
                "question": query_result.question,
                "database": query_result.db_id,
                "predicted_sql": query_result.predicted_sql,
                "extracted_sql": extracted_sql,
                "gold_sql": query_result.gold_sql,
                "llm_response": llm_response.content,
                "exact_match": query_result.exact_match,
                "component_match": query_result.component_match,
                "execution_match": query_result.execution_match,
                "error": query_result.error,
                "performance": {
                    "response_time": llm_response.response_time,
                    "tokens_used": llm_response.usage.get("total_tokens", 0),
                    "model": llm_response.model
                }
            }
            merged_entries.append(merged_entry)

        return {
            "summary": {
                "total_examples": total_examples,
                "exact_match_count": exact_matches,
                "exact_match_accuracy": exact_matches / total_examples if total_examples > 0 else 0.0,
                "component_match_count": component_matches,
                "component_match_accuracy": component_matches / total_examples if total_examples > 0 else 0.0,
                "execution_match_count": execution_matches,
                "execution_match_accuracy": execution_matches / total_examples if total_examples > 0 else 0.0,
                "successful_examples": len(query_results)
            },
            "entries": merged_entries
        }


    async def _handle_multistage_prompting(
        self,
        example: SpiderExample,
        schema,
        strategy: MultiStageStrategy,
        model_params: dict
    ) -> Tuple[Optional[QueryResult], Optional[LLMResponse]]:
        """Handle multi-stage prompting workflow with ablation support."""

        analysis_content = ""
        analysis_response = None

        # Stage 1: Analysis (optional based on strategy config)
        if strategy.use_analysis:
            analysis_prompt = strategy.generate_analysis_prompt(example.question, schema)
            analysis_response = await self.client.complete_async(
                model=model_params["model"],
                prompt=analysis_prompt,
                temperature=strategy.analysis_temp,
                max_tokens=model_params["max_tokens"],
                retry_count=model_params["retry_count"]
            )

            if not analysis_response.success:
                return None, analysis_response

            analysis_content = analysis_response.content

        # Stage 2: SQL Generation
        sql_prompt = strategy.generate_sql_prompt(
            example.question,
            schema,
            analysis_content
        )
        sql_response = await self.client.complete_async(
            model=model_params["model"],
            prompt=sql_prompt,
            temperature=strategy.generation_temp,
            max_tokens=model_params["max_tokens"],
            retry_count=model_params["retry_count"]
        )

        # Debug: Print raw SQL response for failed cases
        if not sql_response.success:
            print(f"\n=== DEBUG: SQL Generation Failed ===")
            print(f"Model: {model_params['model']}")
            print(f"Success: {sql_response.success}")
            print(f"Error: {sql_response.error}")
            print(f"Content: {repr(sql_response.content)}")
            print(f"Usage: {sql_response.usage}")
            print("===================================\n")
            return None, sql_response

        # Extract SQL
        extracted_sql = strategy.extract_sql_from_response(sql_response.content)
        if not extracted_sql:
            print(f"\n=== DEBUG: SQL Extraction Failed ===")
            print(f"Model: {model_params['model']}")
            print(f"Response Success: {sql_response.success}")
            print(f"Response Content: {repr(sql_response.content)}")
            print(f"Content Length: {len(sql_response.content)}")
            print("===================================\n")
            query_result = QueryResult(
                predicted_sql="",
                gold_sql=example.query,
                db_id=example.db_id,
                question=example.question,
                exact_match=False,
                component_match=False,
                execution_match=False,
                error="Could not extract SQL from response"
            )
            return query_result, sql_response

        # Stage 3: Verification (optional based on strategy config)
        verification_response = None
        if strategy.use_verification:
            verification_prompt = strategy.generate_verification_prompt(
                example.question,
                schema,
                extracted_sql
            )
            verification_response = await self.client.complete_async(
                model=model_params["model"],
                prompt=verification_prompt,
                temperature=strategy.verification_temp,
                max_tokens=512,
                retry_count=model_params["retry_count"]
            )

            # If verification suggests changes, use them
            if verification_response.success and "CORRECT" not in verification_response.content.upper():
                refined_sql = strategy.extract_sql_from_response(verification_response.content)
                if refined_sql:
                    extracted_sql = refined_sql

        # Evaluate final prediction
        query_result = self.spider_manager.evaluate_single_query(
            extracted_sql,
            example.query,
            example.db_id,
            example.question
        )

        # Combine responses for tracking (use SQL generation response as primary)
        total_tokens = sql_response.usage.get("total_tokens", 0)
        prompt_tokens = sql_response.usage.get("prompt_tokens", 0)
        completion_tokens = sql_response.usage.get("completion_tokens", 0)
        total_time = sql_response.response_time

        if analysis_response:
            total_tokens += analysis_response.usage.get("total_tokens", 0)
            prompt_tokens += analysis_response.usage.get("prompt_tokens", 0)
            completion_tokens += analysis_response.usage.get("completion_tokens", 0)
            total_time += analysis_response.response_time

        if verification_response:
            total_tokens += verification_response.usage.get("total_tokens", 0)
            prompt_tokens += verification_response.usage.get("prompt_tokens", 0)
            completion_tokens += verification_response.usage.get("completion_tokens", 0)
            total_time += verification_response.response_time

        combined_response = LLMResponse(
            content=sql_response.content,
            model=model_params["model"],
            usage={
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            },
            response_time=total_time,
            success=True
        )

        return query_result, combined_response, extracted_sql

    def _generate_experiment_summary(self) -> Dict[str, Any]:
        """Generate summary of all experiments."""
        return {
            "model_comparison": self.performance_manager.get_model_comparison(),
            "strategy_comparison": self.performance_manager.get_strategy_comparison()
        }

    async def _save_experiment_results(self, results: Dict[str, Any], experiment_name: str) -> None:
        """Save experiment results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f"{experiment_name}_{timestamp}_results.json")

        import json
        with open(filename, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"Results saved to: {filename}")

    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    async def quick_test(
        self,
        model_name: str = "meta-llama/llama-3.2-3b-instruct",
        strategy: str = "multi_stage",
        num_examples: int = 10
    ) -> Dict[str, Any]:
        """Run a quick test with minimal configuration."""
        print(f"Running quick test: {model_name} with {strategy}")

        test_config = {
            "name": "quick_test",
            "models": [model_name],
            "strategy": strategy,
            "max_examples": num_examples,
            "spider_path": "spider",
            "split": "dev"
        }

        return await self.run_experiment(test_config)

    async def compare_models(
        self,
        models: List[str],
        strategy: str = "multi_stage",
        num_examples: int = 50
    ) -> Dict[str, Any]:
        """Quick model comparison."""
        print(f"Comparing models: {', '.join([m.split('/')[-1] for m in models])}")

        comparison_config = {
            "name": "model_comparison",
            "models": models,
            "strategy": strategy,
            "max_examples": num_examples,
            "spider_path": "spider",
            "split": "dev"
        }

        return await self.run_experiment(comparison_config)

    async def compare_strategies(
        self,
        strategies: List[str],
        model: str = "meta-llama/llama-3.2-3b-instruct",
        num_examples: int = 50
    ) -> Dict[str, Any]:
        """Quick strategy comparison."""
        print(f"Comparing strategies: {', '.join(strategies)}")

        strategy_config = {
            "name": "strategy_comparison",
            "models": [model],
            "strategies": strategies,
            "max_examples": num_examples,
            "spider_path": "spider",
            "split": "dev"
        }

        return await self.run_experiment(strategy_config)

    def generate_report(self) -> str:
        """Generate a comprehensive text report."""
        return self.performance_manager.generate_summary_report()

    async def close(self):
        """Clean up resources."""
        await self.client.close()


# Convenience functions for easy usage
async def run_quick_experiment(
    models: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    num_examples: int = 20,
    spider_path: str = "spider"
) -> Dict[str, Any]:
    """Convenience function for running a quick experiment."""
    if not models:
        models = ["meta-llama/llama-3.2-3b-instruct", "deepseek/deepseek-r1"]
    if not strategies:
        strategies = ["multi_stage", "multi_stage_no_verification"]

    testbench = NL2SQLTestbench(spider_path=spider_path)

    try:
        experiment_config = {
            "name": "quick_experiment",
            "models": models,
            "strategies": strategies,
            "max_examples": num_examples,
            "spider_path": spider_path,
            "split": "dev"
        }

        results = await testbench.run_experiment(experiment_config)
        return results
    finally:
        await testbench.close()


def run_from_preset(preset_name: str, spider_path: str = "spider") -> Dict[str, Any]:
    """Run experiment from a configuration preset."""
    from . import config

    preset_configs = {
        "quick": config.get_quick_test_config,
        "model_comparison": config.get_model_comparison_config,
        "strategy_ablation": config.get_strategy_ablation_config,
        "comprehensive": config.get_comprehensive_config
    }

    if preset_name not in preset_configs:
        raise ValueError(f"Unknown preset: {preset_name}")

    experiment_config = preset_configs[preset_name]()
    experiment_config["spider_path"] = spider_path

    testbench = NL2SQLTestbench(spider_path=spider_path)

    async def _run():
        try:
            return await testbench.run_experiment(experiment_config)
        finally:
            await testbench.close()

    return asyncio.run(_run())