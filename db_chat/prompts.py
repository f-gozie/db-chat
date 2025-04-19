"""Prompt templates for LLM interactions in the database chat app."""

import logging

from .helpers import get_dialect_syntax_example

logger = logging.getLogger(__name__)


def get_sql_generation_system_prompt(schema, allowed_tables, db_dialect: str):
    """Return the system prompt for SQL generation.

    Args:
        schema: Database schema information
        allowed_tables: List of allowed table names
        db_dialect: The specific SQL dialect (e.g., 'PostgreSQL', 'MySQL').

    Returns:
        str: Formatted system prompt
    """
    syntax_example = get_dialect_syntax_example(db_dialect)

    return f"""You are a database assistant connected to a {db_dialect} database.
You can only execute read-only SQL queries.

Here is the schema for the tables you have access to:
{schema}

Allowed tables: {', '.join(allowed_tables)}

IMPORTANT INSTRUCTIONS FOR HANDLING QUERIES:
1. Focus DIRECTLY on the user's most recent question - this is the primary task
2. Always pay careful attention to field types and constraints in the schema
3. For VARCHAR fields with CHOICES, preserve exact spacing and capitalization (e.g., 'ON GOING' not 'ONGOING', 'NOT STARTED' not 'NOT_STARTED')
4. Always use exact string values from the CHOICES list, including spaces and special characters
5. Pay attention to primary keys, foreign keys, and table relationships
6. Query only tables listed in the schema - don't reference tables that aren't available
7. You may use Common Table Expressions (CTEs) with WITH clauses to organize complex queries
8. CRITICAL: Ensure all SQL syntax, especially functions (like date/time functions), is compatible with {db_dialect}. {syntax_example}

Your goal is to generate precise SQL queries that retrieve ONLY the specific information requested in the current question, with careful attention to field types, constraints, and exact CHOICE values."""


def get_sql_generation_user_prompt(user_query):
    """Return the user prompt for SQL generation.

    Args:
        user_query: The user's original query

    Returns:
        str: Formatted user prompt
    """
    return f"""Based on my question "{user_query}", generate a single, complete, and executable READ-ONLY SQL query to retrieve the EXACT information I'm asking for.

Follow these guidelines:
1. Your query MUST be complete and executable - do not leave it unfinished
2. Verify there are no trailing commas at the end of column lists
3. Include all necessary JOINs and WHERE clauses
4. Always alias tables when using multiple tables (e.g., "users AS u")
5. The query must end with a semicolon
6. IMPORTANT: Respond with ONLY the SQL query, without any markdown formatting or explanation

If the query cannot be answered with the available tables, respond with EXACTLY: "I cannot answer this query with the available data and tools." and then explain why.

CRITICAL: Make sure your query is DIRECTLY relevant to my specific question. Don't query for loosely related information."""


def get_interpretation_system_prompt(schema, user_query):
    """Return the system prompt for result interpretation.

    Args:
        schema: Database schema information
        user_query: The user's original query

    Returns:
        str: Formatted system prompt
    """
    return f"""You are a database assistant that provides clear, direct answers to user questions based on SQL query results.

When interpreting query results:
1. Focus ONLY on answering the SPECIFIC question that was asked
2. Don't provide tangential information that wasn't requested
3. Keep responses precise and to the point
4. Format numbers clearly when presenting them in your answer
5. Don't mention tables or column names unless necessary for clarity

Here is the schema you're working with:
{schema}

The user's original question was: "{user_query}"
Your job is to answer THIS EXACT question based on the SQL results."""


def get_interpretation_user_prompt(user_query, sql_query, raw_result):
    """Return the user prompt for result interpretation.

    Args:
        user_query: The user's original query
        sql_query: The executed SQL query
        raw_result: The raw result from the SQL query

    Returns:
        str: Formatted user prompt
    """
    return f"""I just ran this SQL query to answer my question "{user_query}":
```sql
{sql_query}
```

The query returned this result:
```
{raw_result}
```

Please provide a direct, focused answer to my specific question based on these results. Keep your response relevant and concise.

If the SQL result indicates an error, explain the issue briefly and suggest how I might rephrase my question."""


def get_error_system_prompt():
    """Return the system prompt for error explanation.

    Returns:
        str: Formatted system prompt
    """
    return """You are a helpful database assistant. Help explain SQL errors in simple terms."""


def get_error_user_prompt(sql_query, error_message):
    """Return the user prompt for error explanation.

    Args:
        sql_query: The SQL query that caused the error
        error_message: The error message from the database

    Returns:
        str: Formatted user prompt
    """
    return f"""I tried to run this SQL query to answer my last question:
```sql
{sql_query}
```

But the query failed with this error:
{error_message}

Please explain the error in simple terms and suggest how I might rephrase my question to get a successful answer. Be brief and helpful, avoiding technical jargon when possible."""


def get_user_friendly_error_prompt(user_query, error_type, error_message):
    """
    Return a user-centric, context-aware error prompt for the LLM to explain errors in relation to the user's question.

    Args:
        user_query: The user's original question
        error_type: A string categorizing the error (e.g., 'invalid_structure', 'security_violation', etc.)
        error_message: The backend/system error message (not shown directly to the user)

    Returns:
        str: A prompt for the LLM to generate a user-friendly error explanation
    """
    base = f"""A user asked the following question:
"{user_query}"

Unfortunately, I couldn't provide a direct answer because of the following issue:
"""
    if error_type == "invalid_structure":
        base += "It looks like the question couldn't be translated into a valid database query. Please try to be more specific or ask about a particular table, field, or value."
    elif error_type == "security_violation":
        base += "The question requested information or actions that aren't allowed for security or privacy reasons. Please focus your question on the available data or tables."
    elif error_type == "cannot_answer":
        base += "The information needed to answer this question isn't available in the current database. Please try rephrasing or asking about something else."
    elif error_type == "trailing_comma":
        base += "There was a technical issue with the generated query. Please try rephrasing your question in a different way."
    elif error_type == "schema_error":
        base += "There was a problem accessing the database structure needed to answer your question. Please try again later or contact support if the issue persists."
    elif error_type == "conversation_creation_error":
        base += "There was a problem starting a new conversation. Please try again or refresh the page."
    elif error_type == "generation_execution_exception":
        base += "An unexpected error occurred while preparing to answer your question. Please try again or rephrase your question."
    elif error_type == "internal_processing_error":
        base += "An unexpected internal error occurred. Please try again or rephrase your question."
    else:
        base += (
            "An error occurred. Please try rephrasing your question or try again later."
        )
    base += "\n\nIf possible, suggest a way the user could rephrase their question to get a better answer. Be friendly, concise, and avoid technical jargon."
    return base
