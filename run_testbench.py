#!/usr/bin/env python3
"""
NL2SQL Testbench CLI - Simplified

Simple command-line interface for running NL2SQL evaluations.
"""

import argparse
import asyncio
import sys
import os
from typing import Optional, List

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.testbench import NL2SQLTestbench
from src.config import AVAILABLE_MODELS


async def run_quick_test(args: argparse.Namespace) -> None:
    """Run a quick test."""
    testbench = NL2SQLTestbench(spider_path=args.spider_path)

    try:
        results = await testbench.quick_test(
            model_name=args.model,
            strategy=args.strategy,
            num_examples=args.examples
        )
        print("Quick test completed successfully!")

    finally:
        await testbench.close()


async def run_model_comparison(args: argparse.Namespace) -> None:
    """Run model comparison."""
    models: List[str] = args.models if args.models else AVAILABLE_MODELS
    testbench = NL2SQLTestbench(spider_path=args.spider_path)

    try:
        results = await testbench.compare_models(
            models=models,
            strategy="multi_stage",
            num_examples=args.examples
        )
        print("Model comparison completed successfully!")

    finally:
        await testbench.close()


async def run_strategy_comparison(args: argparse.Namespace) -> None:
    """Run strategy comparison."""
    strategies: List[str] = args.strategies if args.strategies else [
        "multi_stage", "multi_stage_no_analysis", "multi_stage_no_verification", "multi_stage_simple"
    ]
    testbench = NL2SQLTestbench(spider_path=args.spider_path)

    try:
        results = await testbench.compare_strategies(
            strategies=strategies,
            model=args.model,
            num_examples=args.examples
        )
        print("Strategy comparison completed successfully!")

    finally:
        await testbench.close()


def list_models(args: argparse.Namespace) -> None:
    """List available models."""
    print("Available models:")
    for model in AVAILABLE_MODELS:
        print(f"  - {model}")


def list_strategies(args: argparse.Namespace) -> None:
    """List available strategies."""
    strategies: List[str] = ["multi_stage", "multi_stage_no_analysis", "multi_stage_no_verification", "multi_stage_simple"]
    print("Available strategies:")
    for strategy in strategies:
        print(f"  - {strategy}")


async def run_minimal_test(args: argparse.Namespace) -> None:
    """Run a minimal test with just 3 examples for quick validation."""
    testbench = NL2SQLTestbench(spider_path=args.spider_path)

    try:
        results = await testbench.quick_test(
            model_name="meta-llama/llama-3.1-8b-instruct",
            strategy="multi_stage",
            num_examples=3
        )
        print("Minimal test completed successfully!")
        print(f"Results: {results}")

    finally:
        await testbench.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NL2SQL Testbench - Simplified")
    parser.add_argument("--spider-path", default="spider", help="Path to Spider dataset")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Quick test command
    quick_parser = subparsers.add_parser("quick", help="Run quick test")
    quick_parser.add_argument("--model", default="meta-llama/llama-3.2-3b-instruct", help="Model to test")
    quick_parser.add_argument("--strategy", default="multi_stage", help="Strategy to use")
    quick_parser.add_argument("--examples", type=int, default=10, help="Number of examples")

    # Model comparison command
    models_parser = subparsers.add_parser("compare-models", help="Compare multiple models")
    models_parser.add_argument("--models", nargs="+", help="Models to compare (default: all)")
    models_parser.add_argument("--examples", type=int, default=50, help="Number of examples")

    # Strategy comparison command
    strategies_parser = subparsers.add_parser("compare-strategies", help="Compare multiple strategies")
    strategies_parser.add_argument("--model", default="meta-llama/llama-3.2-3b-instruct", help="Model to use")
    strategies_parser.add_argument("--strategies", nargs="+", help="Strategies to compare (default: all)")
    strategies_parser.add_argument("--examples", type=int, default=50, help="Number of examples")

    # List commands
    subparsers.add_parser("list-models", help="List available models")
    subparsers.add_parser("list-strategies", help="List available strategies")

    # Minimal test command
    minimal_parser = subparsers.add_parser("minimal", help="Run minimal test with 3 examples for quick validation")

    args = parser.parse_args()

    if args.command == "quick":
        asyncio.run(run_quick_test(args))
    elif args.command == "compare-models":
        asyncio.run(run_model_comparison(args))
    elif args.command == "compare-strategies":
        asyncio.run(run_strategy_comparison(args))
    elif args.command == "minimal":
        asyncio.run(run_minimal_test(args))
    elif args.command == "list-models":
        list_models(args)
    elif args.command == "list-strategies":
        list_strategies(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()