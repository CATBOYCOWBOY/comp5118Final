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

    def evaluate_with_test_suite(self, predictions: List[Tuple[str, str]], gold_queries: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Evaluate predictions using test-suite-sql-eval.

        Args:
            predictions: List of (predicted_sql, db_id) tuples
            gold_queries: List of (gold_sql, db_id) tuples

        Returns:
            Dictionary with evaluation results
        """

        # Create temporary files for evaluation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as pred_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as gold_file:

            # Write predictions
            for pred_sql, _ in predictions:
                pred_file.write(pred_sql + '\n')

            # Write gold queries in format: "gold_sql \t db_id"
            for gold_sql, db_id in gold_queries:
                gold_file.write(f"{gold_sql}\t{db_id}\n")

            pred_file_path = pred_file.name
            gold_file_path = gold_file.name

        try:
            # Run test-suite evaluation
            eval_script = "evaluation.py"  # Run from test-suite directory

            # Use absolute paths
            abs_tables_file = os.path.abspath(self.tables_file)
            abs_db_dir = os.path.abspath(self.database_dir)

            cmd = [
                "python3", eval_script,
                "--gold", gold_file_path,
                "--pred", pred_file_path,
                "--etype", "all",  # Both test suite and exact match
                "--db", abs_db_dir,
                "--table", abs_tables_file,
                "--plug_value"  # Plug in gold values for fair comparison
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_suite_path)

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

            return {
                "test_suite_accuracy": test_suite_acc,
                "exact_match_accuracy": exact_match_acc,
                "total_examples": len(predictions),
                "evaluation_output": output
            }

        finally:
            # Clean up temporary files
            try:
                os.unlink(pred_file_path)
                os.unlink(gold_file_path)
            except OSError:
                pass

    def _extract_accuracy(self, output: str, metric_name: str) -> float:
        """Extract accuracy value from test-suite-sql-eval output."""
        lines = output.split('\n')

        # Look for the execution and exact match accuracy sections
        if "execution" in metric_name.lower():
            # Find the execution accuracy line and extract the "all" column value
            for line in lines:
                if line.strip().startswith("execution"):
                    parts = line.split()
                    if len(parts) >= 5:  # Should have values for easy, medium, hard, extra, all
                        try:
                            return float(parts[-1])  # Last value should be "all"
                        except ValueError:
                            continue
        elif "exact" in metric_name.lower() or "match" in metric_name.lower():
            # Find the exact match accuracy line
            for line in lines:
                if line.strip().startswith("exact match"):
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            return float(parts[-1])  # Last value should be "all"
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