import json
import sqlite3
import os
import sys
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class Spider2Example:
    instance_id: str
    db: str
    question: str
    external_knowledge: Optional[str]


@dataclass
class Spider2QueryResult:
    instance_id: str
    question: str
    db: str
    predicted_sql: str
    execution_result: Optional[pd.DataFrame]
    gold_result: Optional[pd.DataFrame]
    execution_match: bool
    error: Optional[str] = None


class Spider2Manager:
    """Spider2-lite dataset manager for text-in-results-out evaluation."""

    def __init__(self, spider2_path: str = "Spider2"):
        self.spider2_path = spider2_path
        self.examples: List[Spider2Example] = []

        # Setup paths
        self.data_file = os.path.join(spider2_path, "spider2-lite", "spider2-lite.jsonl")
        self.db_dir = os.path.join(spider2_path, "spider2-lite", "resource", "databases", "local_sqlite")
        self.docs_dir = os.path.join(spider2_path, "spider2-lite", "resource", "documents")
        self.eval_dir = os.path.join(spider2_path, "spider2-lite", "evaluation_suite")

        self._load_examples()

    def _load_examples(self) -> None:
        """Load Spider2-lite examples from JSONL file."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Spider2-lite data file not found: {self.data_file}")

        with open(self.data_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                # Only load local SQLite examples for now
                if data['instance_id'].startswith('local'):
                    example = Spider2Example(
                        instance_id=data['instance_id'],
                        db=data['db'],
                        question=data['question'],
                        external_knowledge=data.get('external_knowledge')
                    )
                    self.examples.append(example)

    def get_examples(self, max_examples: Optional[int] = None) -> List[Spider2Example]:
        """Get Spider2 examples, optionally limited by count."""
        examples = self.examples
        if max_examples:
            examples = examples[:max_examples]
        return examples

    def get_database_path(self, db_name: str) -> Optional[str]:
        """Get the full path to a SQLite database."""
        # Try different possible filenames
        possible_names = [
            f"{db_name}.sqlite",
            f"{db_name.lower()}.sqlite",
            f"{db_name.upper()}.sqlite"
        ]

        for name in possible_names:
            db_path = os.path.join(self.db_dir, name)
            if os.path.exists(db_path):
                return db_path

        return None

    def get_external_knowledge(self, filename: str) -> Optional[str]:
        """Load external knowledge document if available."""
        if not filename:
            return None

        doc_path = os.path.join(self.docs_dir, filename)
        if os.path.exists(doc_path):
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error loading external knowledge {filename}: {e}")

        return None

    def execute_sql(self, sql: str, db_name: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """Execute SQL query against the specified database."""
        db_path = self.get_database_path(db_name)
        if not db_path:
            return False, None, f"Database not found: {db_name}"

        try:
            conn = sqlite3.connect(db_path)
            # Execute the query and return results as DataFrame
            result_df = pd.read_sql_query(sql, conn)
            conn.close()
            return True, result_df, None
        except Exception as e:
            return False, None, str(e)

    def get_database_schema(self, db_name: str) -> Optional[str]:
        """Get database schema information for prompts."""
        db_path = self.get_database_path(db_name)
        if not db_path:
            return None

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            schema_info = []
            for (table_name,) in tables:
                # Get column information for each table
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()

                col_info = []
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    col_info.append(f"{col_name} ({col_type})")

                schema_info.append(f"Table: {table_name}")
                schema_info.append(f"Columns: {', '.join(col_info)}")
                schema_info.append("")

            conn.close()
            return "\n".join(schema_info)
        except Exception as e:
            print(f"Error getting schema for {db_name}: {e}")
            return None

    def evaluate_query_execution(
        self,
        predicted_sql: str,
        example: Spider2Example
    ) -> Spider2QueryResult:
        """Evaluate a predicted SQL query by executing it and comparing results."""

        # Execute predicted SQL
        success, pred_result, error = self.execute_sql(predicted_sql, example.db)

        # For now, we don't have gold execution results locally
        # In a full implementation, you'd load from evaluation_suite/gold/exec_result
        gold_result = None
        execution_match = False  # Would compare with gold results if available

        if not success:
            execution_match = False

        return Spider2QueryResult(
            instance_id=example.instance_id,
            question=example.question,
            db=example.db,
            predicted_sql=predicted_sql,
            execution_result=pred_result,
            gold_result=gold_result,
            execution_match=execution_match,
            error=error
        )

    def format_prompt_context(self, example: Spider2Example) -> str:
        """Format context for the LLM prompt including schema and external knowledge."""
        context_parts = []

        # Add database schema
        schema = self.get_database_schema(example.db)
        if schema:
            context_parts.append("Database Schema:")
            context_parts.append(schema)

        # Add external knowledge if available
        if example.external_knowledge:
            knowledge = self.get_external_knowledge(example.external_knowledge)
            if knowledge:
                context_parts.append(f"External Knowledge ({example.external_knowledge}):")
                context_parts.append(knowledge)
                context_parts.append("")

        return "\n".join(context_parts)

    def list_available_databases(self) -> List[str]:
        """List all available SQLite databases."""
        if not os.path.exists(self.db_dir):
            return []

        databases = []
        for file in os.listdir(self.db_dir):
            if file.endswith('.sqlite'):
                databases.append(file.replace('.sqlite', ''))

        return sorted(databases)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded Spider2 dataset."""
        stats = {
            "total_examples": len(self.examples),
            "available_databases": len(self.list_available_databases()),
            "examples_with_external_knowledge": len([e for e in self.examples if e.external_knowledge]),
            "database_distribution": {}
        }

        # Count examples per database
        for example in self.examples:
            db_name = example.db
            if db_name not in stats["database_distribution"]:
                stats["database_distribution"][db_name] = 0
            stats["database_distribution"][db_name] += 1

        return stats