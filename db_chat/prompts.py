"""Prompt templates for LLM interactions in the database chat app."""


def get_sql_generation_system_prompt(schema, allowed_tables):
    """Return the system prompt for SQL generation.

    Args:
        schema: Database schema information
        allowed_tables: List of allowed tables

    Returns:
        str: Formatted system prompt
    """
    return f"""You are a database assistant connected to a PostgreSQL database.
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
