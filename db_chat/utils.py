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
    """Check if text has the basic structure of a SQL query using sqlparse."""
    parsed = sqlparse.parse(text)
    if not parsed or not parsed[0].tokens:
        return False
    stmt = parsed[0]
    # Must start with SELECT
    first_token = next((t for t in stmt.tokens if not t.is_whitespace), None)
    if not first_token or not (
        first_token.ttype is DML and first_token.value.upper() == "SELECT"
    ):
        return False
    # Must contain FROM after SELECT
    found_select = False
    found_from = False
    found_table_after_from = False
    for idx, token in enumerate(stmt.tokens):
        if token.ttype is DML and token.value.upper() == "SELECT":
            found_select = True
        if found_select and token.ttype is Keyword and token.value.upper() == "FROM":
            found_from = True
            # Check if there's something after FROM
            next_meaningful_token = next(
                (t for t in stmt.tokens[idx + 1 :] if not t.is_whitespace), None
            )
            if next_meaningful_token:
                found_table_after_from = True
            break
    if not (found_select and found_from and found_table_after_from):
        return False
    # Parentheses must be balanced
    if text.count("(") != text.count(")"):
        return False
    # No double commas
    if ",," in text or re.search(r"\s,\s,", text):
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

    cte_names = []
    if re.search(r"\bwith\b", lowered, re.IGNORECASE):
        cte_matches = re.findall(r"(\w+)\s+as\s*\(", lowered, re.IGNORECASE)
        cte_names = [cte.lower() for cte in cte_matches]
        logger.info(f"Found CTEs in query: {cte_names}")

    mentioned_tables = extract_table_names_with_sqlparse(text)
    if "with" in lowered:
        # Find all FROM ... patterns in the query
        all_from_tables = re.findall(r"\bfrom\s+(\w+)", lowered)
        for table in all_from_tables:
            if table.lower() not in cte_names:
                mentioned_tables.add(table)

    allowed_tables_lower = [t.lower() for t in allowed_tables]
    for table in mentioned_tables:
        if table.lower() not in allowed_tables_lower and table.lower() not in cte_names:
            logger.warning(f"SQL references non-allowed table: {table}")
            return False
    return True
