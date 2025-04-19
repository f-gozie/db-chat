"""Utility functions for the database chat app."""

import logging
import re

import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import DML, Keyword

logger = logging.getLogger(__name__)


def clean_sql_query(text):
    """Extract and clean a SQL query from LLM-generated text."""
    clean_sql = re.sub(r"```sql|```", "", text)
    sql_lines = [line.strip() for line in clean_sql.split("\n") if line.strip()]
    return " ".join(sql_lines)


def is_valid_sql_structure(text):
    """Check if text has the basic structure of a SQL query, including CTEs."""
    text_stripped = text.strip()
    text_upper = text_stripped.upper()

    # Basic checks: balanced parentheses and no double commas
    if text.count("(") != text.count(")"):
        return False
    if ",," in text:
        return False

    # Helper to check non-empty table after FROM/JOIN
    def has_table_after_from(query: str) -> bool:
        match = re.search(r"\bFROM\s+([a-zA-Z0-9_]+)", query, re.IGNORECASE)
        return bool(match and match.group(1).strip())

    # Handle CTEs starting with WITH
    if text_upper.startswith("WITH"):
        # Extract main query after CTE block
        try:
            last_paren = text_stripped.rfind(")")
            main_query = text_stripped[last_paren + 1 :].lstrip(" ;")
        except Exception:
            return False

        if not main_query.upper().startswith("SELECT"):
            return False
        if "FROM" not in main_query.upper():
            return False
        if not has_table_after_from(main_query):
            return False
        return True

    # Regular queries: must start with SELECT and contain FROM
    if not text_upper.startswith("SELECT"):
        return False
    if "FROM" not in text_upper:
        return False
    if not has_table_after_from(text_stripped):
        return False
    return True


def extract_table_names_with_sqlparse(sql):
    """Extract table names from SQL using sqlparse, ignoring columns in expressions."""
    tables = set()
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        from_seen = False
        for token in stmt.tokens:
            if from_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        tables.add(identifier.get_real_name())
                elif isinstance(token, Identifier):
                    tables.add(token.get_real_name())
                elif token.ttype is Keyword:
                    from_seen = False
            if token.ttype is Keyword and token.value.upper() == "FROM":
                from_seen = True
            # Handle JOINs
            if token.ttype is Keyword and "JOIN" in token.value.upper():
                idx = stmt.token_index(token)
                next_token = stmt.token_next(idx, skip_ws=True, skip_cm=True)
                if next_token and isinstance(next_token[1], Identifier):
                    tables.add(next_token[1].get_real_name())

        # Handle tables in CTEs
        if hasattr(stmt, "tokens"):
            for token in stmt.tokens:
                if (
                    token.is_group
                    and token.tokens
                    and token.tokens[0].value.upper() == "WITH"
                ):
                    for cte_token in token.tokens:
                        if hasattr(cte_token, "tokens"):
                            # Look for SELECT statements within the CTE
                            for subtoken in cte_token.tokens:
                                if hasattr(subtoken, "tokens"):
                                    # Recursively extract tables from nested statements
                                    for table in extract_tables_from_token(subtoken):
                                        tables.add(table)

    # Fallback: simple regex to catch tables missed by sqlparse (e.g., deeply nested CTEs)
    # This is not perfect SQL parsing but provides a safety net for common cases.
    regex_tables = re.findall(r"\b(?:from|join)\s+([a-zA-Z0-9_]+)", sql, re.IGNORECASE)
    for tbl in regex_tables:
        tables.add(tbl)
    return tables


def extract_tables_from_token(token):
    """Recursively extract table names from a token and its children."""
    tables = set()
    if not hasattr(token, "tokens"):
        return tables

    from_seen = False
    for subtoken in token.tokens:
        if from_seen:
            if isinstance(subtoken, IdentifierList):
                for identifier in subtoken.get_identifiers():
                    tables.add(identifier.get_real_name())
            elif isinstance(subtoken, Identifier):
                tables.add(subtoken.get_real_name())
            elif subtoken.ttype is Keyword:
                from_seen = False
        if subtoken.ttype is Keyword and subtoken.value.upper() == "FROM":
            from_seen = True
            # Get the next token after FROM which should be the table name
            idx = token.token_index(subtoken)
            next_token = token.token_next(idx, skip_ws=True, skip_cm=True)
            if next_token and isinstance(next_token[1], Identifier):
                tables.add(next_token[1].get_real_name())
            elif next_token and hasattr(next_token[1], "value"):
                # Handle simple table names not wrapped as Identifiers
                tables.add(next_token[1].value)
        # Handle JOINs
        if subtoken.ttype is Keyword and "JOIN" in subtoken.value.upper():
            idx = token.token_index(subtoken)
            next_token = token.token_next(idx, skip_ws=True, skip_cm=True)
            if next_token and isinstance(next_token[1], Identifier):
                tables.add(next_token[1].get_real_name())
            elif next_token and hasattr(next_token[1], "value"):
                # Handle simple table names not wrapped as Identifiers
                tables.add(next_token[1].value)

        # Recursively check nested tokens
        if hasattr(subtoken, "tokens"):
            for table in extract_tables_from_token(subtoken):
                tables.add(table)

    return tables


def is_safe_sql(text, allowed_tables):
    """Check if SQL query is safe and only references allowed tables."""
    lowered = text.lower()
    if re.search(
        r"\b(insert|update|delete|drop|alter|create|truncate)\b", lowered, re.IGNORECASE
    ):
        logger.warning(f"SQL contains write operations: {text}")
        return False

    allowed_tables_lower = [t.lower() for t in allowed_tables]

    cte_names = []
    cte_definitions: dict[str, str] = {}
    if re.search(r"\bwith\b", lowered, re.IGNORECASE):
        # Capture CTE names and their definition body for validation
        cte_pattern = re.compile(
            r"(\w+)\s+as\s*\((.*?)\)\s*(,|with|select|$)", re.IGNORECASE | re.DOTALL
        )
        for match in cte_pattern.finditer(text):
            cte_name = match.group(1).lower()
            cte_body = match.group(2)
            cte_names.append(cte_name)
            cte_definitions[cte_name] = cte_body
        if cte_names:
            logger.info(f"Found CTEs in query: {cte_names}")

    mentioned_tables = extract_table_names_with_sqlparse(text)
    if "with" in lowered:
        # Find all FROM ... patterns in the query (outside of CTE definitions)
        all_from_tables = re.findall(r"\bfrom\s+(\w+)", lowered)
        for table in all_from_tables:
            mentioned_tables.add(table)

        # Validate each CTE definition references only allowed tables
        for cte_name, body in cte_definitions.items():
            inner_tables = extract_table_names_with_sqlparse(body)
            # If CTE body references disallowed tables OR references no tables at all, mark unsafe
            if not inner_tables:
                # Allow empty body if the CTE name itself is an allowed table (e.g., renaming)
                if cte_name not in allowed_tables_lower:
                    logger.warning(
                        f"CTE '{cte_name}' does not reference any tables – flagging as unsafe."
                    )
                    return False
            for t in inner_tables:
                if t.lower() not in allowed_tables_lower and t.lower() not in cte_names:
                    logger.warning(
                        f"CTE '{cte_name}' references disallowed table '{t}'."
                    )
                    return False

    # Remove obvious column names captured from EXTRACT / expressions
    blacklist = {"year", "month", "day", "created", "updated"}
    mentioned_tables = {t for t in mentioned_tables if t.lower() not in blacklist}

    for table in mentioned_tables:
        if table.lower() not in allowed_tables_lower and table.lower() not in cte_names:
            logger.warning(f"SQL references non-allowed table: {table}")
            return False
    return True
