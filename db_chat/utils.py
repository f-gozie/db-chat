"""Utility functions for the database chat app."""

import logging
import re

logger = logging.getLogger(__name__)


def clean_sql_query(text):
    """Clean and extract SQL query from LLM-generated text.

    Args:
        text: Raw text from LLM that may contain a SQL query

    Returns:
        str: Cleaned SQL query with consistent formatting
    """
    # Remove code blocks
    clean_sql = re.sub(r"```sql|```", "", text)

    # Remove leading/trailing whitespace and get the first non-empty block
    sql_lines = [line.strip() for line in clean_sql.split("\n") if line.strip()]
    combined_sql = " ".join(sql_lines)

    return combined_sql


def is_valid_sql_structure(text):
    """Check if text has the basic structure of a SQL query.

    Args:
        text: Text to validate

    Returns:
        bool: True if text has valid SQL structure
    """
    # Convert to lowercase for case-insensitive checking
    text = text.lower()

    # Must have SELECT and FROM in proper order
    if not re.search(r"\bselect\b.*\bfrom\b", text, re.IGNORECASE):
        return False

    # Check for structural balance - parentheses must match
    if text.count("(") != text.count(")"):
        return False

    # Basic syntax check - no SQL syntax errors like double commas
    if re.search(r",,", text) or re.search(r"\s,\s,", text):
        return False

    # Check for proper table reference
    # Should have "from table" or "from table as alias" pattern
    if not re.search(r"\bfrom\b\s+\w+(\s+\b(as)?\b\s+\w+)?", text, re.IGNORECASE):
        return False

    return True


def is_safe_sql(text, allowed_tables):
    """Check if SQL query is safe and only references allowed tables.

    Args:
        text: SQL query to check
        allowed_tables: List of allowed table names

    Returns:
        bool: True if SQL query is safe
    """
    text = text.lower()

    # Check for write operations (should only be SELECT for read-only)
    if re.search(
        r"\b(insert|update|delete|drop|alter|create|truncate)\b", text, re.IGNORECASE
    ):
        logger.warning(f"SQL contains write operations: {text}")
        return False

    # Extract CTE names to exclude them from table validation
    cte_names = []

    # Handle WITH clause with multiple CTEs
    if re.search(r"\bwith\b", text, re.IGNORECASE):
        # Find all CTE definitions in the format "name AS (" including across multiple lines
        cte_matches = re.findall(r"(\w+)\s+as\s*\(", text, re.IGNORECASE)
        cte_names = [cte.lower() for cte in cte_matches]
        logger.info(f"Found CTEs in query: {cte_names}")

    # Extract table names from the main query and all subqueries
    # This will find tables in FROM clauses (including subqueries)
    from_clauses = re.findall(
        r"\bfrom\s+(\w+)(?:\s+(?:as\s+)?(\w+))?", text, re.IGNORECASE
    )

    # Find tables in JOIN clauses (including in subqueries)
    join_clauses = re.findall(
        r"\b(?:inner|left|right|outer|cross)?\s*join\s+(\w+)(?:\s+(?:as\s+)?(\w+))?",
        text,
        re.IGNORECASE,
    )

    # Find tables in subqueries within WHERE, HAVING, or other clauses
    subquery_clauses = re.findall(
        r"\(\s*select\b.*?\bfrom\s+(\w+)(?:\s+(?:as\s+)?(\w+))?",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    # Combine all table references
    mentioned_tables = []
    for match in from_clauses + join_clauses + subquery_clauses:
        if isinstance(match, tuple) and len(match) > 0:
            table_name = match[0]
            mentioned_tables.append(table_name)
        elif isinstance(match, str):
            mentioned_tables.append(match)

    # Check if all mentioned tables are in the allowed tables list or are CTEs
    allowed_tables_lower = [t.lower() for t in allowed_tables]
    for table in mentioned_tables:
        if table.lower() not in allowed_tables_lower and table.lower() not in cte_names:
            logger.warning(f"SQL references non-allowed table: {table}")
            return False

    return True
