from typing import List, Optional
from dataclasses import dataclass
import re

from spider1 import Spider1Example, Spider1Manager


@dataclass
class PromptResult:
    prompt: str
    sql_response: str
    error: Optional[str] = None


class Spider1Strategy:
    def __init__(self, spider1_manager: Spider1Manager, **kwargs):
        self.spider1_manager = spider1_manager
        self.strategy_options = kwargs

    def generate_prompt(self, example: Spider1Example) -> str:
        context = self.spider1_manager.format_prompt_context(example)

        prompt = f"""You are an expert SQL analyst. Utilize these stages to answer the question:

        Stage 1: Analyze the question and identify key requirements
        Stage 2: Examine the database schema and identify relevant tables and columns
        Stage 3: Plan the SQL query structure (joins, filters, aggregations, etc.)
        Stage 4: Generate the final SQL query

        {context}

        Question: {example.question}

        Requirements:
        - Use proper SQL syntax for SQLite
        - Do NOT include any prefixes like "sql:", "sqlite:", or "query:" before your SQL
        - Return ONLY the final SQL query without code blocks or additional formatting
        - Do not wrap the query in backticks or markdown formatting


        Final SQL Query:"""

        return prompt

    def process_prompt_to_sql(self, llm_response: str) -> Optional[str]:
        """Process LLM response and extract SQL query."""
        return self.extract_sql_from_response(llm_response)

    @staticmethod
    def extract_sql_from_response(response: str) -> Optional[str]:
        """Extract SQL query from LLM response, handling prefix-free responses."""
        response = response.strip()

        # Remove common prefixes if they exist
        prefixes = [
            r'^(?:sql|sqlite|query)\s*:\s*',
            r'^(?:final\s+)?(?:sql\s+)?query\s*:\s*',
            r'^(?:the\s+)?(?:sql\s+)?(?:query\s+)?(?:is\s*)?:\s*'
        ]
        for prefix in prefixes:
            response = re.sub(prefix, '', response, flags=re.IGNORECASE)

        # Handle code blocks
        sql_pattern = r'```(?:sql)?\s*(.*?)\s*```'
        matches = re.findall(sql_pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

        # Look for SQL starting keywords
        lines = response.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
                sql_lines = [line]
                for next_line in lines[i + 1:]:
                    next_line = next_line.strip()
                    if not next_line or next_line.startswith(('Question:', 'Answer:', 'Explanation:', 'Stage')):
                        break
                    sql_lines.append(next_line)
                return ' '.join(sql_lines).rstrip(';').strip()

        # Try to match complete SQL statements
        select_match = re.search(r'((?:SELECT|WITH).*?)(?:\n\n|\n[A-Z]|$)', response, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()

        # If response looks like it's just SQL (starts with SELECT, WITH, etc.), return it directly
        response_clean = response.strip().rstrip(';')
        if response_clean.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
            return response_clean

        return None