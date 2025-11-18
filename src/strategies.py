from typing import List, Optional
from dataclasses import dataclass
import re

from .spider import SpiderExample, DatabaseSchema, SchemaFormatter


@dataclass
class PromptResult:
    prompt: str
    sql_response: str
    extracted_sql: Optional[str] = None
    error: Optional[str] = None


class MultiStageStrategy:
    """Multi-stage prompting: analyze → generate → verify."""

    def __init__(self,
                 use_analysis: bool = True,
                 use_verification: bool = True,
                 analysis_temp: float = 0.2,
                 generation_temp: float = 0.1,
                 verification_temp: float = 0.05):
        self.use_analysis = use_analysis
        self.use_verification = use_verification
        self.analysis_temp = analysis_temp
        self.generation_temp = generation_temp
        self.verification_temp = verification_temp

    def generate_analysis_prompt(self, question: str, schema: DatabaseSchema) -> str:
        """Stage 1: Analyze the question and schema."""
        schema_text = SchemaFormatter.format_schema_for_prompt(schema)

        return f"""Analyze the following natural language question and database schema to understand what SQL query is needed.

        {schema_text}

        Question: {question}

        Please identify:
        1. Which tables need to be accessed?
        2. What columns should be selected?
        3. What conditions/filters are needed?
        4. Are any JOINs required?
        5. Are aggregations needed?
        6. Is sorting or limiting required?

        Provide your analysis:"""

    def generate_sql_prompt(
        self,
        question: str,
        schema: DatabaseSchema,
        analysis: str,
        examples: Optional[List[SpiderExample]] = None
    ) -> str:
        """Stage 2: Generate SQL based on analysis."""
        schema_text = SchemaFormatter.format_schema_for_prompt(schema)

        prompt = f"Based on the following analysis, generate the SQL query:\n\n"
        prompt += f"Analysis: {analysis}\n\n"
        prompt += f"{schema_text}\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Generate the SQL query:"

        return prompt

    def generate_verification_prompt(
        self,
        question: str,
        schema: DatabaseSchema,
        sql_query: str
    ) -> str:
        """Stage 3: Verify and potentially refine the SQL."""
        return f"""Review the following SQL query for correctness:

        Question: {question}
        SQL Query: {sql_query}

        Check for:
        1. Syntax errors
        2. Correct table and column names
        3. Logical correctness
        4. Efficiency

        If the query is correct, respond with "CORRECT". If not, provide the corrected version:"""

    def process_prompt_to_sql(self, llm_response: str) -> Optional[str]:
        """Process LLM response and extract SQL query for evaluation."""
        return self.extract_sql_from_response(llm_response)

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