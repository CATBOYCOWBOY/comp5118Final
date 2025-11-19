from typing import List, Optional
from dataclasses import dataclass
import re

from spider2 import Spider2Example, Spider2Manager


@dataclass
class PromptResult:
    prompt: str
    sql_response: str
    extracted_sql: Optional[str] = None
    error: Optional[str] = None


class Spider2Strategy:
    """Strategy for Spider2-lite text-in-results-out evaluation."""

    def __init__(self, spider2_manager: Spider2Manager, **kwargs):
        self.spider2_manager = spider2_manager
        self.strategy_options = kwargs

    def generate_prompt(self, example: Spider2Example) -> str:
        """Generate a prompt for Spider2 example."""
        # Get database context
        context = self.spider2_manager.format_prompt_context(example)

        prompt = f"""You are an expert SQL analyst. Generate a SQL query to answer the given question.

{context}

Question: {example.question}

Requirements:
- Write a precise SQL query that answers the question
- Use proper SQL syntax for SQLite
- Consider any external knowledge provided
- Return ONLY the SQL query without additional explanation

SQL Query:"""

        return prompt

    def process_prompt_to_sql(self, llm_response: str) -> Optional[str]:
        """Process LLM response and extract SQL query."""
        return self.extract_sql_from_response(llm_response)

    def evaluate_result(self, predicted_sql: str, example: Spider2Example):
        """Evaluate the SQL by executing it and checking results."""
        return self.spider2_manager.evaluate_query_execution(predicted_sql, example)

    @staticmethod
    def extract_sql_from_response(response: str) -> Optional[str]:
        """Extract SQL query from LLM response."""
        sql_pattern = r'```(?:sql)?\s*(.*?)\s*```'
        matches = re.findall(sql_pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
                sql_lines = [line]
                for next_line in lines[lines.index(line) + 1:]:
                    next_line = next_line.strip()
                    if not next_line or next_line.startswith(('Question:', 'Answer:', 'Explanation:')):
                        break
                    sql_lines.append(next_line)
                return ' '.join(sql_lines).rstrip(';').strip()
        select_match = re.search(r'(SELECT.*?)(?:\n\n|\n[A-Z]|$)', response, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()

        return None