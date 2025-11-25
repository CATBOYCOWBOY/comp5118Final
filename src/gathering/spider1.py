"""
Spider1 dataset manager for traditional Text2SQL evaluation with test-suite-sql-eval.
"""
import json
import os
import sqlite3
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import tempfile
from datetime import datetime
import re
import logging

# Import the normalization function
from prompting_strategy import Spider1Strategy


@dataclass
class Spider1Example:
    instance_id: str
    db_id: str
    question: str
    query: str
    query_toks: List[str]
    question_toks: List[str]


@dataclass
class Spider1EvalResult:
    instance_id: str
    question: str
    db_id: str
    predicted_sql: str
    gold_sql: str
    test_suite_accuracy: Optional[float]
    exact_match_accuracy: Optional[float]
    error: Optional[str] = None


class Spider1Manager:
    """Spider1 dataset manager for text-to-SQL evaluation with test-suite-sql-eval."""

    def __init__(self, spider_path: str = "spider", test_suite_path: str = "test-suite-sql-eval"):
        self.spider_path = spider_path
        self.test_suite_path = test_suite_path
        self.examples: List[Spider1Example] = []

        # Setup paths
        self.dev_file = os.path.join(spider_path, "evaluation_examples", "examples", "dev.json")
        self.train_file = os.path.join(spider_path, "evaluation_examples", "examples", "train_spider.json")
        self.tables_file = os.path.join(spider_path, "evaluation_examples", "examples", "tables.json")
        self.database_dir = os.path.join(test_suite_path, "database")

        # Load table information
        self.tables_info = self._load_tables_info()

        self._load_examples()

    def _load_tables_info(self) -> Dict[str, Any]:
        """Load database table information."""
        if not os.path.exists(self.tables_file):
            raise FileNotFoundError(f"Spider tables file not found: {self.tables_file}")

        with open(self.tables_file, 'r') as f:
            tables_data = json.load(f)

        # Convert to dict keyed by db_id for quick lookup
        tables_dict = {}
        for table_info in tables_data:
            tables_dict[table_info['db_id']] = table_info

        return tables_dict

    def _load_examples(self, split: str = "dev") -> None:
        """Load Spider1 examples from JSON file."""
        data_file = self.dev_file if split == "dev" else self.train_file

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Spider data file not found: {data_file}")

        with open(data_file, 'r') as f:
            data = json.load(f)

        for i, item in enumerate(data):
            example = Spider1Example(
                instance_id=f"{item['db_id']}_{i}",
                db_id=item['db_id'],
                question=item['question'],
                query=item['query'],
                query_toks=item.get('query_toks', []),
                question_toks=item.get('question_toks', [])
            )
            self.examples.append(example)

    def get_examples(self, max_examples: Optional[int] = None) -> List[Spider1Example]:
        """Get Spider1 examples, optionally limited by count."""
        examples = self.examples
        if max_examples:
            examples = examples[:max_examples]
        return examples

    def format_prompt_context(self, example: Spider1Example) -> str:
        """Format database schema context for prompting."""
        if example.db_id not in self.tables_info:
            return f"Database: {example.db_id}\nNo schema information available."

        table_info = self.tables_info[example.db_id]

        context = f"Database: {example.db_id}\n\nSchema:\n"

        # Format tables and columns
        table_names = table_info['table_names_original']
        column_names = table_info['column_names_original']
        column_types = table_info['column_types']

        # Group columns by table
        tables_with_columns = {}
        for i, (table_idx, column_name) in enumerate(column_names):
            if table_idx == -1:  # Skip special cases
                continue

            table_name = table_names[table_idx]
            if table_name not in tables_with_columns:
                tables_with_columns[table_name] = []

            column_type = column_types[i] if i < len(column_types) else "unknown"
            tables_with_columns[table_name].append(f"{column_name} ({column_type})")

        # Format output
        for table_name, columns in tables_with_columns.items():
            context += f"\nTable: {table_name}\n"
            context += "Columns: " + ", ".join(columns) + "\n"

        # Add foreign keys if available
        if 'foreign_keys' in table_info and table_info['foreign_keys']:
            context += "\nForeign Keys:\n"
            for fk in table_info['foreign_keys']:
                if len(fk) >= 2 and fk[0] < len(column_names) and fk[1] < len(column_names):
                    col1_info = column_names[fk[0]]
                    col2_info = column_names[fk[1]]
                    if col1_info[0] >= 0 and col2_info[0] >= 0:
                        table1 = table_names[col1_info[0]]
                        table2 = table_names[col2_info[0]]
                        context += f"- {table1}.{col1_info[1]} → {table2}.{col2_info[1]}\n"

        return context

    def create_results_directory(self, base_run_dir: str, model_name: str, strategy_name: str) -> str:
        """Create a structured results directory for a specific model/strategy run."""
        model_safe = model_name.split('/')[-1].replace('-', '_')
        strategy_safe = strategy_name.replace('-', '_')
        results_dir = os.path.join(base_run_dir, f"{model_safe}_{strategy_safe}")
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    def save_predictions_to_file(self, predictions: List[Tuple[str, str]], output_file: str) -> None:
        """Save predictions to a text file, one per line."""
        with open(output_file, 'w') as f:
            for pred_sql, _ in predictions:
                sanitized = Spider1Strategy.normalize_sql(pred_sql) if pred_sql else ""
                f.write((sanitized or "SELECT 1") + '\n')

    def evaluate_with_test_suite(
        self,
        predictions: List[Tuple[str, str]],
        gold_queries: List[Tuple[str, str]],
        results_dir: Optional[str] = None,
        plug_value: bool = True,
        keep_distinct: bool = False,
        etype: str = "all"
    ) -> Dict[str, Any]:
        """
        Evaluate predictions using test-suite-sql-eval.

        Args:
            predictions: List of (predicted_sql, db_id) tuples
            gold_queries: List of (gold_sql, db_id) tuples
            results_dir: Optional results directory to save files to
            plug_value: Whether to plug gold values into predictions (recommended when models don't emit values)
            keep_distinct: Preserve DISTINCT keywords during evaluation
            etype: Evaluation type passed to test-suite-sql-eval (exec, match, or all)

        Returns:
            Dictionary with evaluation results
        """

        logger = logging.getLogger("Spider1Manager")
        if len(predictions) != len(gold_queries):
            raise ValueError(f"Prediction/gold length mismatch: {len(predictions)} predictions vs {len(gold_queries)} gold")

        # If results_dir is provided, save files there, otherwise use temp files
        if results_dir:
            pred_file_path = os.path.join(results_dir, "predictions.txt")
            gold_file_path = os.path.join(results_dir, "gold.txt")
            abs_pred_file_path = os.path.abspath(pred_file_path)
            abs_gold_file_path = os.path.abspath(gold_file_path)

            # Save predictions file
            self.save_predictions_to_file(predictions, pred_file_path)

            # Save gold file
            with open(gold_file_path, 'w') as gold_file:
                for gold_sql, db_id in gold_queries:
                    gold_file.write(f"{gold_sql.strip()}\t{db_id}\n")

            cleanup_files = False
        else:
            # Create temporary files for evaluation (fallback)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as pred_file, \
                 tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as gold_file:

                # Write predictions (normalized for better exact matching)
                self.save_predictions_to_file(predictions, pred_file.name)

                # Write gold queries in format: "gold_sql \t db_id" (keep original format)
                for gold_sql, db_id in gold_queries:
                    gold_file.write(f"{gold_sql.strip()}\t{db_id}\n")

                pred_file_path = pred_file.name
                gold_file_path = gold_file.name
                abs_pred_file_path = pred_file_path
                abs_gold_file_path = gold_file_path

            cleanup_files = True

        try:
            # Run test-suite evaluation
            eval_script = "evaluation.py"  # Run from test-suite directory

            # Use absolute paths
            abs_tables_file = os.path.abspath(self.tables_file)
            abs_db_dir = os.path.abspath(self.database_dir)

            cmd = [
                "python3", eval_script,
                "--gold", abs_gold_file_path,
                "--pred", abs_pred_file_path,
                "--etype", etype,
                "--db", abs_db_dir
            ]

            if etype in ("all", "match"):
                cmd.extend(["--table", abs_tables_file])
            if plug_value:
                cmd.append("--plug_value")
            if keep_distinct:
                cmd.append("--keep_distinct")

            # Add timeout to prevent hanging on problematic SQL queries (e.g., badly formatted JOINs)
            # Use 30 seconds per example with a 30-minute maximum cap
            timeout_seconds = min(1800, len(predictions) * 30)  # 30 min max, 30 sec per example
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_suite_path, timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                logger.error("Evaluation timed out after %d seconds with %d examples. Likely problematic SQL queries causing hangs.", timeout_seconds, len(predictions))
                return {
                    "error": f"Evaluation timed out after {timeout_seconds}s - likely problematic SQL queries",
                    "test_suite_accuracy": 0.0,
                    "exact_match_accuracy": 0.0
                }
            logger.info("test-suite eval exited code %s; stdout len=%d stderr len=%d", result.returncode, len(result.stdout), len(result.stderr))

            if result.returncode != 0:
                return {
                    "error": f"Test suite evaluation failed: {result.stderr}",
                    "test_suite_accuracy": 0.0,
                    "exact_match_accuracy": 0.0
                }

            # Parse output for accuracies
            output = result.stdout
            test_suite_acc = self._extract_accuracy(output, "execution")
            exact_match_acc = self._extract_accuracy(output, "exact match")

            # Save evaluation output if results_dir is provided
            if results_dir:
                eval_output_file = os.path.join(results_dir, "evaluation_output.txt")
                with open(eval_output_file, 'w') as f:
                    f.write(output)

            return {
                "test_suite_accuracy": test_suite_acc,
                "exact_match_accuracy": exact_match_acc,
                "total_examples": len(predictions),
                "evaluation_output": output
            }

        finally:
            # Clean up temporary files (only if using temp files)
            if cleanup_files:
                try:
                    os.unlink(pred_file_path)
                    os.unlink(gold_file_path)
                except OSError:
                    pass

    def _extract_accuracy(self, output: str, metric_name: str) -> float:
        """Extract accuracy value from test-suite-sql-eval output."""
        target_prefix = "execution" if "execution" in metric_name.lower() else "exact match"
        for line in output.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith(target_prefix):
                numbers = re.findall(r"\d+\.\d+|\d+", line)
                if numbers:
                    try:
                        return float(numbers[-1])
                    except ValueError:
                        continue
        return 0.0

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the Spider1 dataset."""
        db_counts = {}
        for example in self.examples:
            db_counts[example.db_id] = db_counts.get(example.db_id, 0) + 1

        return {
            "total_examples": len(self.examples),
            "total_databases": len(db_counts),
            "database_distribution": db_counts
        }
