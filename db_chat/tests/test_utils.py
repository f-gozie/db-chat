import pytest

from db_chat import utils


class TestUtils:
    def test_clean_sql_query_removes_code_blocks(self):
        sql = "```sql\nSELECT * FROM users;\n```"
        assert utils.clean_sql_query(sql) == "SELECT * FROM users;"

    def test_clean_sql_query_handles_empty(self):
        assert utils.clean_sql_query("") == ""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("SELECT * FROM users", True),
            ("select * from users", True),
            ("SELECT name FROM users WHERE id = 1", True),
            ("SELECT * users", False),
            ("SELECT * FROM", False),
            ("SELECT * FROM users WHERE (id = 1", False),
            ("SELECT * FROM users,,", False),
            ("SELECT * FROM users as u", True),
            ("SELECT * FROM users u", True),
            ("SELECT * FROM", False),
        ],
    )
    def test_is_valid_sql_structure(self, text, expected):
        assert utils.is_valid_sql_structure(text) == expected

    def test_is_safe_sql_blocks_write_ops(self):
        allowed = ["users"]
        assert not utils.is_safe_sql("INSERT INTO users VALUES (1)", allowed)
        assert not utils.is_safe_sql('UPDATE users SET name = "a"', allowed)
        assert not utils.is_safe_sql("DELETE FROM users", allowed)
        assert not utils.is_safe_sql("DROP TABLE users", allowed)
        assert not utils.is_safe_sql("ALTER TABLE users", allowed)
        assert not utils.is_safe_sql("TRUNCATE users", allowed)

    def test_is_safe_sql_allows_only_allowed_tables(self):
        allowed = ["users", "projects"]
        assert utils.is_safe_sql("SELECT * FROM users", allowed)
        assert not utils.is_safe_sql("SELECT * FROM admins", allowed)
        assert utils.is_safe_sql(
            "SELECT * FROM users JOIN projects ON users.id = projects.user_id", allowed
        )
        assert not utils.is_safe_sql(
            "SELECT * FROM users JOIN admins ON users.id = admins.user_id", allowed
        )
