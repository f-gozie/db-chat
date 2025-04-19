import pytest

from db_chat import utils


class DummyToken:
    def __init__(self, tokens=None, value=None, ttype=None, is_group=False):
        self.tokens = tokens or []
        self.value = value
        self.ttype = ttype
        self.is_group = is_group

    def token_index(self, token):
        return 0

    def token_next(self, idx, skip_ws=True, skip_cm=True):
        return (None, DummyToken(value="not_identifier"))


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
            ("SELECT * FROM ", False),
            ("SELECT * FROM users WHERE (id = 1", False),
            ("SELECT * FROM users,,", False),
            ("SELECT * FROM users as u", True),
            ("SELECT * FROM users u", True),
            ("WITH cte AS (SELECT * FROM users) SELECT * FROM", False),
            ("WITH cte AS (SELECT * FROM users) SELECT * FROM ", False),
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

    def test_is_safe_sql_handles_ctes(self):
        allowed = ["users", "projects"]
        cte_query = """
        WITH user_counts AS (
            SELECT user_id, COUNT(*) as count
            FROM projects
            GROUP BY user_id
        )
        SELECT users.name, user_counts.count
        FROM users
        JOIN user_counts ON users.id = user_counts.user_id
        """
        assert utils.is_safe_sql(cte_query, allowed)

        # Test multiple CTEs
        multi_cte_query = """
        WITH
            project_counts AS (SELECT COUNT(*) FROM projects),
            user_stats AS (SELECT id, name FROM users)
        SELECT * FROM user_stats, project_counts
        """
        assert utils.is_safe_sql(multi_cte_query, allowed)

        # Test multiple CTEs with one referencing another
        nested_cte_query = """
        WITH
            total_items AS (SELECT COUNT(*) as total FROM projects),
            item_breakdown AS (
                SELECT
                    status,
                    COUNT(*) as count,
                    (COUNT(*) * 100.0 / (SELECT total FROM total_items)) as percentage
                FROM projects
                GROUP BY status
            )
        SELECT status, count, percentage
        FROM item_breakdown
        ORDER BY count DESC
        """
        assert utils.is_safe_sql(nested_cte_query, allowed)

        # Test with invalid table in the CTE
        invalid_cte_query = """
        WITH stats AS (
            SELECT * FROM admins
        )
        SELECT * FROM stats
        """
        assert not utils.is_safe_sql(invalid_cte_query, allowed)

    def test_is_safe_sql_does_not_flag_column_in_expression(self):
        allowed = ["projects_project"]
        sql = "SELECT AVG(completion_rate) AS average_completion_rate FROM projects_project WHERE EXTRACT(YEAR FROM created) = 2023;"
        assert utils.is_safe_sql(sql, allowed)

    def test_extract_table_names_with_sqlparse_empty(self):
        assert utils.extract_table_names_with_sqlparse("") == set()

    def test_extract_table_names_with_sqlparse_malformed(self):
        # Should not raise, just return empty set
        assert utils.extract_table_names_with_sqlparse("SELECT FROM") == set()

    def test_extract_tables_from_token_empty(self):
        class Dummy:
            pass

        assert utils.extract_tables_from_token(Dummy()) == set()

    def test_is_safe_sql_malformed(self):
        # Should not raise, just return False for write op
        assert not utils.is_safe_sql("DROP TABLE", ["users"])
        # Should return False for non-allowed table
        assert not utils.is_safe_sql("SELECT * FROM forbidden", ["users"])
        # Should return True for allowed table
        assert utils.is_safe_sql("SELECT * FROM users", ["users"])

    def test_extract_table_names_with_sqlparse_malformed_sql(self):
        # Should not raise, just return empty set
        assert utils.extract_table_names_with_sqlparse("SELECT FROM") == set()
        assert utils.extract_table_names_with_sqlparse("nonsense") == set()

    def test_extract_table_names_with_sqlparse_nested_cte(self):
        sql = """
        WITH outer_cte AS (
            WITH inner_cte AS (SELECT * FROM users) SELECT * FROM inner_cte
        )
        SELECT * FROM outer_cte
        """
        tables = utils.extract_table_names_with_sqlparse(sql)
        assert "users" in tables
        assert "inner_cte" in tables or "outer_cte" in tables

    def test_extract_tables_from_token_no_tokens(self):
        dummy = DummyToken()
        assert utils.extract_tables_from_token(dummy) == set()

    def test_extract_tables_from_token_nested(self):
        # Nested tokens with FROM and JOIN
        join_token = DummyToken(value="JOIN", ttype=utils.Keyword)
        from_token = DummyToken(value="FROM", ttype=utils.Keyword)
        id_token = DummyToken(value="users")
        parent = DummyToken(tokens=[from_token, id_token, join_token], is_group=True)
        # Patch token_next to return id_token for FROM and JOIN
        parent.token_next = lambda idx, skip_ws=True, skip_cm=True: (None, id_token)
        parent.token_index = lambda token: 0
        tables = utils.extract_tables_from_token(parent)
        assert "users" in tables

    def test_is_safe_sql_cte_shadowing(self):
        allowed = ["users"]
        sql = "WITH users AS (SELECT 1) SELECT * FROM users"
        assert utils.is_safe_sql(sql, allowed)
        sql2 = "WITH admins AS (SELECT 1) SELECT * FROM users, admins"
        assert not utils.is_safe_sql(sql2, allowed)

    def test_is_safe_sql_table_name_substring(self):
        allowed = ["users"]
        sql = "SELECT * FROM users1"
        assert not utils.is_safe_sql(sql, allowed)
        sql2 = "SELECT * FROM users"
        assert utils.is_safe_sql(sql2, allowed)
