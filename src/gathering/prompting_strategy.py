from __future__ import annotations

from typing import List, Optional, Iterable, TYPE_CHECKING
from dataclasses import dataclass
import re

if TYPE_CHECKING:
    from spider1 import Spider1Example, Spider1Manager
else:
    Spider1Example = Spider1Manager = object


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
        """Extract SQL query from an LLM response in a way that matches test-suite expectations."""
        if not response:
            return None

        response = response.strip()

        # 1) Prefer fenced code blocks if present
        fenced_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            cleaned = Spider1Strategy._clean_sql_candidate(fenced_match.group(1))
            if cleaned:
                return cleaned

        # 2) Check for inline backticks (single-line answers)
        inline_match = re.search(r"`([^`]+)`", response)
        if inline_match:
            cleaned = Spider1Strategy._clean_sql_candidate(inline_match.group(1))
            if cleaned:
                return cleaned

        # 3) Scan line-by-line for the first SQL-looking statement and collect contiguous SQL lines.
        lines = response.splitlines()
        for idx, raw_line in enumerate(lines):
            stripped = Spider1Strategy._strip_leading_prefixes(raw_line)
            if Spider1Strategy._looks_like_sql_start(stripped):
                sql_lines = [stripped]
                for follow in lines[idx + 1:]:
                    follow_clean = follow.strip()
                    if Spider1Strategy._is_termination_line(follow_clean):
                        break
                    sql_lines.append(follow_clean)
                cleaned = Spider1Strategy._clean_sql_candidate(" ".join(sql_lines))
                if cleaned:
                    return cleaned

        # 4) Regex fallback to capture the first SQL-looking span.
        select_match = re.search(r"((?:WITH|SELECT|INSERT|UPDATE|DELETE)\s.+)", response, re.IGNORECASE | re.DOTALL)
        if select_match:
            cleaned = Spider1Strategy._clean_sql_candidate(select_match.group(1))
            if cleaned:
                return cleaned

        # 5) If the whole response is SQL, return as-is after cleaning.
        if Spider1Strategy._looks_like_sql_start(response):
            cleaned = Spider1Strategy._clean_sql_candidate(response)
            if cleaned:
                return cleaned

        return None

    @staticmethod
    def _clean_sql_candidate(sql: str) -> Optional[str]:
        """Normalize a candidate SQL string without destroying casing or values."""
        if not sql:
            return None

        sql = sql.strip().strip('`').strip()
        sql = Spider1Strategy._strip_leading_prefixes(sql)

        # Trim trailing semicolons and collapse whitespace
        sql = sql.rstrip(';').strip()
        sql = re.sub(r"\s+", " ", sql)

        return sql if sql and Spider1Strategy._looks_like_sql_start(sql) else None

    @staticmethod
    def _looks_like_sql_start(text: str) -> bool:
        return bool(re.match(r"^(WITH|SELECT|INSERT|UPDATE|DELETE)\b", text.strip(), flags=re.IGNORECASE))

    @staticmethod
    def _strip_leading_prefixes(text: str) -> str:
        """Remove common verbal prefixes before SQL."""
        prefixes = (
            r"^(?:sql|sqlite|query)\s*:\s*",
            r"^(?:final\s+)?(?:sql\s+)?query\s*:\s*",
            r"^(?:the\s+)?(?:sql\s+)?(?:query\s+)?(?:is\s*)?:\s*",
            r"^(?:answer|final answer)\s*:\s*",
        )
        cleaned = text.strip()
        for prefix in prefixes:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _is_termination_line(text: str) -> bool:
        """Lines that indicate the SQL snippet has ended."""
        if not text:
            return True
        termination_tokens: Iterable[str] = ("question:", "answer:", "explanation:", "stage", "final")
        lowered = text.lower()
        return any(lowered.startswith(tok) for tok in termination_tokens)

    @staticmethod
    def normalize_sql(sql: str) -> str:
        """Normalize SQL for better exact matching by handling case and spacing."""
        if not sql:
            return ""

        # Remove extra whitespace and normalize spacing without altering casing/values
        sql = re.sub(r"\s+", " ", sql).strip()
        sql = re.sub(r"\s*,\s*", ", ", sql)
        sql = sql.rstrip(";").strip()

        return sql
