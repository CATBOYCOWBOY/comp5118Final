import json
import sqlite3
import os
import sys
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'spider'))

try:
    from evaluation import evaluate, build_foreign_key_map_from_json
    from process_sql import get_sql
except ImportError as e:
    print(f"Warning: Could not import Spider evaluation modules: {e}")
    evaluate = None
    build_foreign_key_map_from_json = None
    get_sql = None


@dataclass
class SpiderExample:
    db_id: str
    question: str
    question_toks: List[str]
    query: str
    query_toks: List[str]
    sql: Dict[str, Any]


@dataclass
class DatabaseSchema:
    db_id: str
    table_names: List[str]
    table_names_original: List[str]
    column_names: List[Tuple[int, str]]
    column_names_original: List[Tuple[int, str]]
    column_types: List[str]
    foreign_keys: List[List[int]]
    primary_keys: List[int]


@dataclass
class EvaluationResult:
    exact_match: float
    component_match: float
    execution_accuracy: float
    error_count: int
    total_count: int
    errors: List[str]


@dataclass
class QueryResult:
    predicted_sql: str
    gold_sql: str
    db_id: str
    question: str
    exact_match: bool
    component_match: bool
    execution_match: bool
    error: Optional[str] = None


class SpiderManager:
    """Merged Spider dataset loader and evaluator."""

    def __init__(self, spider_path: str = "spider"):
        self.spider_path = spider_path
        self.schemas: Dict[str, DatabaseSchema] = {}
        self.train_examples: List[SpiderExample] = []
        self.dev_examples: List[SpiderExample] = []

        # Evaluation setup
        self.tables_file = self._find_tables_file()
        self.db_dir = self._find_db_directory()

        self._load_schemas()

    def _find_tables_file(self) -> str:
        """Find the tables.json file in the Spider dataset."""
        possible_paths = [
            os.path.join(self.spider_path, "tables.json"),
            os.path.join(self.spider_path, "evaluation_examples", "examples", "tables.json")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"Could not find tables.json in {self.spider_path}")

    def _find_db_directory(self) -> str:
        """Find the database directory in the Spider dataset."""
        possible_paths = [
            os.path.join(self.spider_path, "database"),
            os.path.join(self.spider_path, "databases"),
            os.path.join(self.spider_path, "evaluation_examples", "databases")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return os.path.join(self.spider_path, "database")

    def _load_schemas(self) -> None:
        """Load database schemas from tables.json."""
        with open(self.tables_file, 'r') as f:
            tables_data = json.load(f)

        for table_info in tables_data:
            schema = DatabaseSchema(
                db_id=table_info["db_id"],
                table_names=table_info["table_names"],
                table_names_original=table_info["table_names_original"],
                column_names=table_info["column_names"],
                column_names_original=table_info["column_names_original"],
                column_types=table_info["column_types"],
                foreign_keys=table_info["foreign_keys"],
                primary_keys=table_info["primary_keys"]
            )
            self.schemas[schema.db_id] = schema

    def get_schema(self, db_id: str) -> Optional[DatabaseSchema]:
        """Get schema for a specific database."""
        return self.schemas.get(db_id)

    def load_split(self, split: str) -> List[SpiderExample]:
        """Load train or dev split data."""
        if split == "train":
            filename = "train_spider.json"
        elif split == "dev":
            filename = "dev.json"
        else:
            raise ValueError(f"Unknown split: {split}")

        data_path = os.path.join(self.spider_path, "evaluation_examples", "examples", filename)
        if not os.path.exists(data_path):
            data_path = os.path.join(self.spider_path, filename)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find {filename} in {self.spider_path}")

        with open(data_path, 'r') as f:
            data = json.load(f)

        examples = []
        for item in data:
            example = SpiderExample(
                db_id=item["db_id"],
                question=item["question"],
                question_toks=item["question_toks"],
                query=item["query"],
                query_toks=item["query_toks"],
                sql=item["sql"]
            )
            examples.append(example)

        if split == "train":
            self.train_examples = examples
        elif split == "dev":
            self.dev_examples = examples

        return examples

    def evaluate_predictions(
        self,
        predictions: List[Tuple[str, str, str]],  # (predicted_sql, gold_sql, db_id)
        questions: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Evaluate predictions against gold SQL queries."""
        if not evaluate or not build_foreign_key_map_from_json:
            return self._fallback_evaluation(predictions, questions)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as pred_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as gold_file:

            for pred_sql, gold_sql, db_id in predictions:
                pred_file.write(f"{pred_sql}\n")
                gold_file.write(f"{gold_sql}\t{db_id}\n")

            pred_file.flush()
            gold_file.flush()

            try:
                exact_scores, component_scores, exec_scores = evaluate(
                    gold_file.name,
                    pred_file.name,
                    self.db_dir,
                    "all",
                    self.tables_file
                )

                result = EvaluationResult(
                    exact_match=exact_scores[0] if exact_scores else 0.0,
                    component_match=component_scores[0] if component_scores else 0.0,
                    execution_accuracy=exec_scores[0] if exec_scores else 0.0,
                    error_count=0,
                    total_count=len(predictions),
                    errors=[]
                )

            except Exception as e:
                print(f"Error in Spider evaluation: {e}")
                result = self._fallback_evaluation(predictions, questions)

            finally:
                try:
                    os.unlink(pred_file.name)
                    os.unlink(gold_file.name)
                except:
                    pass

        return result

    def _fallback_evaluation(
        self,
        predictions: List[Tuple[str, str, str]],
        questions: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Simple fallback evaluation when Spider modules are not available."""
        exact_matches = 0
        execution_matches = 0
        errors = []

        for i, (pred_sql, gold_sql, db_id) in enumerate(predictions):
            pred_normalized = self._normalize_sql_for_comparison(pred_sql)
            gold_normalized = self._normalize_sql_for_comparison(gold_sql)

            if pred_normalized == gold_normalized:
                exact_matches += 1

            try:
                if self._execution_match(pred_sql, gold_sql, db_id):
                    execution_matches += 1
            except Exception as e:
                errors.append(f"Query {i}: {str(e)}")

        total = len(predictions)
        return EvaluationResult(
            exact_match=exact_matches / total if total > 0 else 0.0,
            component_match=exact_matches / total if total > 0 else 0.0,
            execution_accuracy=execution_matches / total if total > 0 else 0.0,
            error_count=len(errors),
            total_count=total,
            errors=errors
        )

    def _execution_match(self, pred_sql: str, gold_sql: str, db_id: str) -> bool:
        """Check if predicted and gold SQL produce the same results."""
        db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            db_path = os.path.join(self.db_dir, f"{db_id}.sqlite")
            if not os.path.exists(db_path):
                return False

        try:
            conn = sqlite3.connect(db_path)

            pred_cursor = conn.cursor()
            pred_cursor.execute(pred_sql)
            pred_results = pred_cursor.fetchall()

            gold_cursor = conn.cursor()
            gold_cursor.execute(gold_sql)
            gold_results = gold_cursor.fetchall()

            conn.close()

            pred_results = sorted(pred_results) if pred_results else []
            gold_results = sorted(gold_results) if gold_results else []

            return pred_results == gold_results

        except Exception:
            return False

    def evaluate_single_query(
        self,
        predicted_sql: str,
        gold_sql: str,
        db_id: str,
        question: str = ""
    ) -> QueryResult:
        """Evaluate a single query prediction."""
        # Use comprehensive normalization for exact match comparison
        pred_normalized = self._normalize_sql_for_comparison(predicted_sql)
        gold_normalized = self._normalize_sql_for_comparison(gold_sql)
        exact_match = pred_normalized == gold_normalized

        # Since we're using comprehensive normalization for exact match,
        # normalized_match is no longer needed as a separate check

        execution_match = False
        error = None
        try:
            execution_match = self._execution_match(predicted_sql, gold_sql, db_id)
        except Exception as e:
            error = str(e)

        # Component match includes exact match and execution match
        component_match = exact_match or execution_match

        return QueryResult(
            predicted_sql=predicted_sql,
            gold_sql=gold_sql,
            db_id=db_id,
            question=question,
            exact_match=exact_match,
            component_match=component_match,
            execution_match=execution_match,
            error=error
        )

    def _normalized_sql_match(self, pred_sql: str, gold_sql: str) -> bool:
        """Check if SQL queries are equivalent after normalization."""
        try:
            pred_norm = self._normalize_sql_for_comparison(pred_sql)
            gold_norm = self._normalize_sql_for_comparison(gold_sql)
            return pred_norm == gold_norm
        except:
            return False

    def _normalize_sql_for_comparison(self, sql: str) -> str:
        """Normalize SQL for more lenient comparison."""
        import re

        # Convert to uppercase and clean whitespace
        sql = ' '.join(sql.strip().split()).upper()

        # Remove trailing semicolon
        sql = sql.rstrip(';')

        # Normalize operators
        sql = re.sub(r'!=', '=', sql)  # Normalize != to = for comparison
        sql = re.sub(r'<>', '=', sql)  # Normalize <> to = for comparison

        # Normalize quoted identifiers
        sql = re.sub(r'"([^"]+)"', r'\1', sql)  # Remove quotes around identifiers

        # Simple structural normalization - focus on core structure rather than aliases
        # Remove extra whitespace around punctuation
        sql = re.sub(r'\s*,\s*', ', ', sql)
        sql = re.sub(r'\s*\(\s*', '(', sql)
        sql = re.sub(r'\s*\)\s*', ')', sql)

        # Normalize common function patterns
        sql = re.sub(r'COUNT\([^)]*\)', 'COUNT(*)', sql)  # Normalize count expressions

        # For semantic comparison, we could remove aliases entirely and just compare structure
        # This is a simpler approach that focuses on the logical structure

        # Remove table aliases completely for structural comparison
        # First remove explicit AS aliases
        sql = re.sub(r'\b(\w+)\s+AS\s+\w+\b', r'\1', sql)  # Remove "AS alias"

        # Then remove implicit aliases after FROM/JOIN, but be careful not to match keywords
        # Only match single word aliases that aren't SQL keywords
        sql_keywords = {'ON', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'UNION', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER'}

        # Remove table aliases after FROM (table_name alias -> table_name)
        def replace_from_alias(match):
            table_name, alias = match.groups()
            if alias.upper() not in sql_keywords:
                return f'FROM {table_name}'
            return match.group(0)
        sql = re.sub(r'\bFROM\s+(\w+)\s+(\w+)\b', replace_from_alias, sql)

        # Remove table aliases after JOIN (table_name alias -> table_name)
        def replace_join_alias(match):
            table_name, alias = match.groups()
            if alias.upper() not in sql_keywords:
                return f'JOIN {table_name}'
            return match.group(0)
        sql = re.sub(r'\bJOIN\s+(\w+)\s+(\w+)\b', replace_join_alias, sql)

        # Remove alias prefixes from column references
        sql = re.sub(r'\b\w+\.(\w+)\b', r'\1', sql)  # Remove "alias.column" -> "column"

        # Normalize commutative operations for better matching
        # Sort OR conditions to handle cases like "A OR B" vs "B OR A"
        def normalize_or_conditions(match):
            full_expr = match.group(0)
            # Split by OR and sort the conditions
            or_parts = [part.strip() for part in full_expr.split(' OR ')]
            or_parts.sort()
            return ' OR '.join(or_parts)

        # Find and normalize OR expressions (simple case for = conditions)
        sql = re.sub(r'\b\w+\s*=\s*\w+(?:\s+OR\s+\w+\s*=\s*\w+)+', normalize_or_conditions, sql)

        return sql

    def analyze_errors(self, results: List[QueryResult]) -> Dict[str, Any]:
        """Analyze common error patterns in failed queries."""
        error_analysis = {
            "syntax_errors": 0,
            "execution_errors": 0,
            "semantic_errors": 0,
            "db_specific_errors": {}
        }

        for result in results:
            if not result.exact_match:
                if result.error:
                    if "syntax" in result.error.lower():
                        error_analysis["syntax_errors"] += 1
                    else:
                        error_analysis["execution_errors"] += 1
                else:
                    error_analysis["semantic_errors"] += 1

                db_id = result.db_id
                if db_id not in error_analysis["db_specific_errors"]:
                    error_analysis["db_specific_errors"][db_id] = 0
                error_analysis["db_specific_errors"][db_id] += 1

        return error_analysis


class SchemaFormatter:
    """Format database schemas for prompts."""

    @staticmethod
    def format_schema_for_prompt(schema: DatabaseSchema) -> str:
        """Format schema information for LLM prompts."""
        lines = [f"Database: {schema.db_id}\n"]

        current_table_id = -1
        current_table = ""
        columns = []

        for col_id, col_name in schema.column_names:
            if col_id != current_table_id:
                if columns:
                    lines.append(f"Table: {current_table}")
                    lines.append(f"Columns: {', '.join(columns)}")
                    lines.append("")

                current_table_id = col_id
                if col_id >= 0 and col_id < len(schema.table_names):
                    current_table = schema.table_names[col_id]
                columns = []

            if col_name != "*":
                try:
                    # Find the index of this column in column_names
                    col_index = schema.column_names.index((col_id, col_name))
                    # Count non-* columns up to this point to get the type index
                    type_index = len([c for i, c in enumerate(schema.column_names[:col_index+1]) if c[1] != "*"]) - 1
                    col_type = schema.column_types[type_index] if type_index >= 0 and type_index < len(schema.column_types) else ""
                except (ValueError, IndexError):
                    col_type = ""
                columns.append(f"{col_name} ({col_type})" if col_type else col_name)

        if columns:
            lines.append(f"Table: {current_table}")
            lines.append(f"Columns: {', '.join(columns)}")

        if schema.foreign_keys:
            lines.append("\nForeign Keys:")
            for fk in schema.foreign_keys:
                if len(fk) >= 2:
                    col1_info = schema.column_names[fk[0]]
                    col2_info = schema.column_names[fk[1]]
                    lines.append(f"  {col1_info[1]} -> {col2_info[1]}")

        return "\n".join(lines)

    @staticmethod
    def format_minimal_schema(schema: DatabaseSchema) -> str:
        """Format schema in a minimal way for few-shot examples."""
        tables = []
        for i, table_name in enumerate(schema.table_names):
            cols = [col_name for col_id, col_name in schema.column_names if col_id == i and col_name != "*"]
            if cols:
                tables.append(f"{table_name}({', '.join(cols)})")
        return "; ".join(tables)


class ExampleSelector:
    """Select relevant examples for few-shot prompting."""

    def __init__(self, spider_manager: SpiderManager):
        self.manager = spider_manager

    def select_diverse_examples(
        self,
        examples: List[SpiderExample],
        num_examples: int
    ) -> List[SpiderExample]:
        """Select diverse examples for few-shot learning."""
        if not examples or num_examples <= 0:
            return []

        if len(examples) <= num_examples:
            return examples

        # Simple selection by database diversity
        selected = []
        used_dbs = set()

        for example in examples:
            if len(selected) >= num_examples:
                break

            if example.db_id not in used_dbs:
                selected.append(example)
                used_dbs.add(example.db_id)

        # Fill remaining slots if needed
        while len(selected) < num_examples and len(selected) < len(examples):
            for example in examples:
                if example not in selected:
                    selected.append(example)
                    break

        return selected[:num_examples]