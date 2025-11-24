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
    print(f"Running Spider1 test with {args.model}")

    results = await testbench.run_test(
        model_name=args.model,
        max_examples=args.examples,
        evaluate=args.evaluate
    )
    print("Test completed successfully!")


async def show_dataset_info(args: argparse.Namespace) -> None:
    """Show dataset information."""
    testbench = Spider1Testbench(spider_path=args.spider_path, test_suite_path=args.test_suite_path)
    stats = testbench.get_dataset_info()


async def run_evaluate_run(args: argparse.Namespace) -> None:
    """Recompute metrics for an existing run directory."""
    testbench = Spider1Testbench(spider_path=args.spider_path, test_suite_path=args.test_suite_path)
    experiments = testbench.evaluate_run(args.run_dir)
    print(f"Evaluation updated for run: {args.run_dir}")
    for key, exp in experiments.items():
        summary = exp.get("summary", {})
        print(f"{key}: exec={summary.get('test_suite_accuracy')} exact={summary.get('exact_match_accuracy')}")


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

    # Run test command
    run_parser = subparsers.add_parser('run_test', help='Run test with single model')
    run_parser.add_argument(
        '--model',
        required=True,
        help='Model to test'
    )
    run_parser.add_argument(
        '--examples',
        type=int,
        default=10,
        help='Number of examples to test (default: 10)'
    )
    run_parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Compute execution/exact-match immediately (default: generation only)'
    )

    # Re-evaluate an existing run
    eval_parser = subparsers.add_parser('evaluate_run', help='Recompute metrics for an existing run directory')
    eval_parser.add_argument(
        '--run-dir',
        required=True,
        help='Path to a run directory under results/'
    )

    # Dataset info command
    info_parser = subparsers.add_parser('info', help='Show dataset information')

    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command == 'run_test':
        asyncio.run(run_quick_test(args))
    elif args.command == 'evaluate_run':
        asyncio.run(run_evaluate_run(args))
    elif args.command == 'info':
        asyncio.run(show_dataset_info(args))
    elif args.command == 'list-models':
        list_models()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
