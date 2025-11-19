#!/usr/bin/env python3
"""
Spider Testbench CLI

Command-line interface for running Spider1 and Spider2 NL2SQL evaluations.
"""

import argparse
import asyncio
import sys
import os
from typing import Optional, List

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.gathering.spider1_testbench import Spider1Testbench
from src.gathering.config import AVAILABLE_MODELS


async def run_quick_test(args: argparse.Namespace) -> None:
    """Run a quick test."""
    testbench = Spider1Testbench(spider_path=args.spider_path, test_suite_path=args.test_suite_path)
    print(f"Running Spider1 quick test with {args.model}")

    results = await testbench.quick_test(
        model_name=args.model,
        max_examples=args.examples
    )
    print("Quick test completed successfully!")


async def run_model_comparison(args: argparse.Namespace) -> None:
    """Run model comparison."""
    models: List[str] = args.models if args.models else AVAILABLE_MODELS[:3]  # Limit to 3 for speed

    testbench = Spider1Testbench(spider_path=args.spider_path, test_suite_path=args.test_suite_path)
    print(f"Running Spider1 model comparison with {len(models)} models")

    results = await testbench.compare_models(
        models=models,
        max_examples=args.examples
    )
    print("Model comparison completed successfully!")


async def show_dataset_info(args: argparse.Namespace) -> None:
    """Show dataset information."""
    testbench = Spider1Testbench(spider_path=args.spider_path, test_suite_path=args.test_suite_path)
    stats = testbench.get_dataset_info()


def list_models() -> None:
    """List available models."""
    print("Available Models:")
    print("=" * 40)
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"{i:2d}. {model}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spider1 NL2SQL Testbench with test-suite-sql-eval",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global arguments
    parser.add_argument(
        "--spider-path",
        default="spider",
        help="Path to Spider1 dataset (default: spider)"
    )
    parser.add_argument(
        "--test-suite-path",
        default="test-suite-sql-eval",
        help="Path to test-suite-sql-eval (default: test-suite-sql-eval)"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Quick test command
    quick_parser = subparsers.add_parser('quick', help='Run quick test with single model')
    quick_parser.add_argument(
        '--model',
        required=True,
        help='Model to test'
    )
    quick_parser.add_argument(
        '--examples',
        type=int,
        default=10,
        help='Number of examples to test (default: 10)'
    )

    # Model comparison command
    compare_parser = subparsers.add_parser('compare-models', help='Compare multiple models')
    compare_parser.add_argument(
        '--models',
        nargs='+',
        help='Models to compare (default: first 3 available models)'
    )
    compare_parser.add_argument(
        '--examples',
        type=int,
        default=50,
        help='Number of examples per model (default: 50)'
    )

    # Dataset info command
    info_parser = subparsers.add_parser('info', help='Show dataset information')

    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command == 'quick':
        asyncio.run(run_quick_test(args))
    elif args.command == 'compare-models':
        asyncio.run(run_model_comparison(args))
    elif args.command == 'info':
        asyncio.run(show_dataset_info(args))
    elif args.command == 'list-models':
        list_models()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()