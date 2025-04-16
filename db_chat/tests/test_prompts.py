import pytest

import db_chat.prompts as prompts


def test_get_sql_generation_system_prompt():
    schema = "table info"
    allowed = ["users", "projects"]
    result = prompts.get_sql_generation_system_prompt(schema, allowed)
    assert "table info" in result
    assert "users" in result and "projects" in result


def test_get_sql_generation_user_prompt():
    user_query = "How many users?"
    result = prompts.get_sql_generation_user_prompt(user_query)
    assert user_query in result
    assert "SQL query" in result


def test_get_interpretation_system_prompt():
    schema = "table info"
    user_query = "How many users?"
    result = prompts.get_interpretation_system_prompt(schema, user_query)
    assert schema in result
    assert user_query in result


def test_get_interpretation_user_prompt():
    user_query = "How many users?"
    sql_query = "SELECT COUNT(*) FROM users;"
    raw_result = "| count |\n| 10 |"
    result = prompts.get_interpretation_user_prompt(user_query, sql_query, raw_result)
    assert user_query in result
    assert sql_query in result
    assert raw_result in result


def test_get_error_system_prompt():
    result = prompts.get_error_system_prompt()
    assert "helpful database assistant" in result


def test_get_error_user_prompt():
    sql_query = "SELECT * FROM users;"
    error_message = "syntax error"
    result = prompts.get_error_user_prompt(sql_query, error_message)
    assert sql_query in result
    assert error_message in result
