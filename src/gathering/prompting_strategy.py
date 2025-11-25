from __future__ import annotations

from typing import Optional, Iterable, TYPE_CHECKING
from dataclasses import dataclass
import re
import logging

if TYPE_CHECKING:
    from spider1 import Spider1Example, Spider1Manager


@dataclass
class PromptResult:
    prompt: str
    sql_response: str
    error: Optional[str] = None


class Spider1Strategy:
    def __init__(self, spider1_manager: 'Spider1Manager', **kwargs):
        self.spider1_manager = spider1_manager
        self.strategy_options = kwargs
        self.logger = logging.getLogger("Spider1Strategy")

    def generate_prompt(self, example: 'Spider1Example') -> str:
        context = self.spider1_manager.format_prompt_context(example)

        prompt = f"""You are an expert SQL analyst. Generate a SQL query to answer the question based on the provided database schema.

        {context}

        Question: {example.question}

        Requirements:
        - Use proper SQL syntax for SQLite
        - Return ONLY the final SQL query
        - Do NOT include any explanations, reasoning, or additional text
        - Do NOT wrap the query in backticks, code blocks, or markdown formatting
        - Do NOT include prefixes like "sql:", "query:", or similar

        SQL Query:"""

        return prompt

    def process_prompt_to_sql(self, llm_response: str) -> Optional[str]:
        """Process LLM response and extract SQL query."""
        extracted_sql = self.extract_sql_from_response(llm_response)

        # Log SQL extraction details
        if extracted_sql:
            self.logger.info("Successfully extracted SQL: %s", extracted_sql[:100] + "..." if len(extracted_sql) > 100 else extracted_sql)
        else:
            self.logger.warning("Failed to extract SQL from response (length=%d): %s",
                              len(llm_response), llm_response[:200] + "..." if len(llm_response) > 200 else llm_response)

        return extracted_sql

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
                logging.getLogger("Spider1Strategy").debug("Extracted SQL from fenced code block: %s", cleaned)
                return cleaned

        # 2) Check for inline backticks (single-line answers)
        inline_match = re.search(r"`([^`]+)`", response)
        if inline_match:
            cleaned = Spider1Strategy._clean_sql_candidate(inline_match.group(1))
            if cleaned:
                logging.getLogger("Spider1Strategy").debug("Extracted SQL from inline backticks: %s", cleaned)
                return cleaned

        # 3) Look for SQL statements that are clearly isolated on their own lines
        lines = response.splitlines()
        for idx, raw_line in enumerate(lines):
            stripped = Spider1Strategy._strip_leading_prefixes(raw_line.strip())
            if Spider1Strategy._looks_like_sql_start(stripped):
                # Check if this line looks like a complete SQL query (not reasoning)
                if Spider1Strategy._looks_like_complete_sql(stripped):
                    cleaned = Spider1Strategy._clean_sql_candidate(stripped)
                    if cleaned:
                        return cleaned

                # Otherwise collect contiguous SQL lines
                sql_lines = [stripped]
                for follow in lines[idx + 1:]:
                    follow_clean = follow.strip()
                    if Spider1Strategy._is_termination_line(follow_clean) or Spider1Strategy._looks_like_reasoning(follow_clean):
                        break
                    if follow_clean:  # Only add non-empty lines
                        sql_lines.append(follow_clean)

                # Only return if we have what looks like a proper SQL query
                full_sql = " ".join(sql_lines)
                if Spider1Strategy._looks_like_complete_sql(full_sql):
                    cleaned = Spider1Strategy._clean_sql_candidate(full_sql)
                    if cleaned:
                        logging.getLogger("Spider1Strategy").debug("Extracted SQL from line scan: %s", cleaned)
                        return cleaned

        # 4) Try to find the last complete SQL statement (models sometimes give answer at end)
        sql_matches = list(re.finditer(r"((?:WITH|SELECT|INSERT|UPDATE|DELETE)\s[^.]*?)(?=\n\n|\.|$)", response, re.IGNORECASE | re.MULTILINE))
        if sql_matches:
            # Try the last match first (often the final answer)
            for match in reversed(sql_matches):
                candidate = match.group(1).strip()
                if Spider1Strategy._looks_like_complete_sql(candidate):
                    cleaned = Spider1Strategy._clean_sql_candidate(candidate)
                    if cleaned:
                        logging.getLogger("Spider1Strategy").debug("Extracted SQL from multi-match (reversed): %s", cleaned)
                        return cleaned

        # 5) Fallback: look for any SQL statement
        select_match = re.search(r"((?:WITH|SELECT|INSERT|UPDATE|DELETE)\s.+?)(?=\n[A-Z]|\n\n|\.|$)", response, re.IGNORECASE | re.DOTALL)
        if select_match:
            cleaned = Spider1Strategy._clean_sql_candidate(select_match.group(1))
            if cleaned and Spider1Strategy._looks_like_complete_sql(cleaned):
                logging.getLogger("Spider1Strategy").debug("Extracted SQL from fallback regex: %s", cleaned)
                return cleaned

        # 6) If the whole response is SQL, return as-is after cleaning.
        if Spider1Strategy._looks_like_sql_start(response) and Spider1Strategy._looks_like_complete_sql(response):
            cleaned = Spider1Strategy._clean_sql_candidate(response)
            if cleaned:
                logging.getLogger("Spider1Strategy").debug("Extracted SQL from whole response: %s", cleaned)
                return cleaned

        logging.getLogger("Spider1Strategy").debug("No SQL found in response")
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
        termination_tokens: Iterable[str] = ("question:", "answer:", "explanation:", "stage", "final", "note:", "but ", "however", "wait")
        lowered = text.lower()
        return any(lowered.startswith(tok) for tok in termination_tokens)

    @staticmethod
    def _looks_like_reasoning(text: str) -> bool:
        """Check if text looks like reasoning rather than SQL."""
        if not text:
            return False
        reasoning_patterns = (
            r"^(?:but|however|wait|hmm|so|maybe|perhaps|alternatively|actually)\b",
            r"\bwhat if\b", r"\bhow do i\b", r"\bthat would be\b",
            r"\bi need to\b", r"\blet me\b", r"\bthe question\b"
        )
        lowered = text.lower()
        return any(re.search(pattern, lowered) for pattern in reasoning_patterns)

    @staticmethod
    def _looks_like_complete_sql(text: str) -> bool:
        """Check if text looks like a complete SQL query rather than reasoning."""
        if not text or not Spider1Strategy._looks_like_sql_start(text):
            return False

        # Strip to handle spacing
        text = text.strip()

        # Must be more than just "WITH" or "SELECT" - needs actual content
        if len(text.split()) < 3:
            return False

        # Should not contain reasoning language mixed with SQL
        reasoning_in_sql = (
            r"\b(?:but how|wait|hmm|actually|however)\b",
            r"\blet me think\b", r"\bwhat about\b", r"\bi think\b",
            r"\bmaybe the\b", r"\bperhaps\b", r"\bthat would\b"
        )
        lowered = text.lower()
        if any(re.search(pattern, lowered) for pattern in reasoning_in_sql):
            return False

        # Should look like it has SQL components (table names, keywords, etc.)
        sql_components = r"\b(?:FROM|WHERE|JOIN|GROUP BY|ORDER BY|HAVING|LIMIT|COUNT|SUM|AVG|MAX|MIN|DISTINCT)\b"
        if not re.search(sql_components, text, re.IGNORECASE):
            # Allow simple queries without these keywords
            if not re.match(r"^(?:WITH|SELECT)\s+\w+", text, re.IGNORECASE):
                return False

        return True

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
