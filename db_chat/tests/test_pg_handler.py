import importlib
from unittest.mock import MagicMock, patch

from db_chat.connectors.postgres.pg_connector import PgConnector
from db_chat.connectors.postgres.pg_handler import PgHandler


# Mock the connector methods used by the handler
@patch("db_chat.connectors.postgres.pg_connector.PgConnector.get_connection")
class TestPgHandler:
    def test_execute_query_success(self, mock_get_connection):
        """Test successful query execution via the handler."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.description = [
            ("id",),
            ("name",),
        ]
        mock_cursor.fetchall.return_value = [{"id": 1, "name": "test"}]
        mock_cursor.execute.side_effect = [None, None, None]  # BEGIN, SELECT, COMMIT

        connector = PgConnector("fake_conn_str")
        handler = PgHandler(connector)

        result = handler.execute_query("SELECT * FROM test")

        mock_get_connection.assert_called_once()
        mock_conn.cursor.assert_called_once()
        assert mock_cursor.execute.call_count == 3
        assert "| id | name |" in result
        assert "| 1 | test |" in result
        assert "1 row(s) returned" in result
        assert "Error" not in result

    def test_execute_query_no_results(self, mock_get_connection):
        """Test query execution with no results returned."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.description = None
        mock_cursor.execute.side_effect = [None, None, None]

        connector = PgConnector("fake_conn_str")
        handler = PgHandler(connector)

        result = handler.execute_query("SELECT * FROM test WHERE 1=0")

        assert (
            result == "Query executed successfully, but it did not return any results."
        )
        assert mock_cursor.execute.call_count == 2

    def test_execute_query_connection_error(self, mock_get_connection):
        """Test query execution when connection fails."""
        mock_get_connection.return_value = None

        connector = PgConnector("fake_conn_str")
        handler = PgHandler(connector)

        result = handler.execute_query("SELECT * FROM test")

        assert "Error: Failed to get database connection" in result
        mock_get_connection.assert_called_once()

    def test_execute_query_execution_error(self, mock_get_connection):
        """Test query execution when the SQL execution raises an error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.execute.side_effect = [
            None,
            Exception("bad sql"),
            None,
        ]

        connector = PgConnector("fake_conn_str")
        handler = PgHandler(connector)

        result = handler.execute_query("SELECT invalid stuff")

        assert "Error: SQL execution failed - bad sql" in result

    @patch("db_chat.connectors.postgres.pg_handler.get_registry")
    def test_get_schema_success_with_registry(
        self, mock_get_registry, mock_get_connection
    ):
        """Test get_schema successfully using model registry and DB fallback."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        mock_registry_instance = MagicMock()
        mock_get_registry.return_value = mock_registry_instance
        mock_registry_instance.get_table_schema.side_effect = (
            lambda table: f"Schema for {table} from Registry"
            if table == "table_from_registry"
            else None
        )

        table_not_in_registry = "table_from_db"
        mock_cursor.execute.side_effect = [
            # Columns query for table_from_db
            None,  # cursor.execute(GET_COLUMNS_QUERY...)
            # PK query
            None,
            # FK query
            None,
            # Check constraints query
            None,
        ]
        # fetchall results corresponding to the execute calls
        mock_cursor.fetchall.side_effect = [
            [("id", "integer", "NO", None), ("name", "text", "YES", None)],  # Columns
            [("id",)],  # PKs
            [],  # FKs
            [],  # Check constraints
        ]

        connector = PgConnector("fake_conn_str")
        handler = PgHandler(connector)

        tables_to_fetch = ["table_from_registry", table_not_in_registry]
        result = handler.get_schema(tables_to_fetch)

        mock_get_registry.assert_called_once()
        assert mock_registry_instance.get_table_schema.call_count == 2
        assert "Schema for table_from_registry from Registry" in result
        assert f"Table: {table_not_in_registry} (Primary Keys: id)" in result
        assert "- id: integer NOT NULL" in result
        assert "- name: text NULL" in result
        assert "Error" not in result

    @patch(
        "db_chat.connectors.postgres.pg_handler.get_registry",
        side_effect=ImportError("No registry"),
    )
    def test_get_schema_fallback_to_db(
        self, mock_get_registry_import_error, mock_get_connection
    ):
        """Test get_schema falling back to DB only when registry import fails."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        table_name = "some_table"
        mock_cursor.execute.side_effect = [None] * 4  # Columns, PK, FK, Checks
        mock_cursor.fetchall.side_effect = [
            [("col1", "varchar", "NO", None)],  # Columns
            [("col1",)],  # PKs
            [],  # FKs
            [],  # Check constraints
        ]

        connector = PgConnector("fake_conn_str")
        handler = PgHandler(connector)
        result = handler.get_schema([table_name])

        assert f"Table: {table_name} (Primary Keys: col1)" in result
        assert "- col1: varchar NOT NULL" in result
        assert "Error" not in result
        mock_get_registry_import_error.assert_called_once()

    def test_get_schema_db_error(self, mock_get_connection):
        """Test get_schema when direct DB fetch fails."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        mock_cursor.execute.side_effect = Exception("db schema fetch failed")

        connector = PgConnector("fake_conn_str")
        handler = PgHandler(connector)

        with patch(
            "db_chat.connectors.postgres.pg_handler.get_registry"
        ) as mock_get_reg:
            mock_registry_instance = MagicMock()
            mock_get_reg.return_value = mock_registry_instance
            mock_registry_instance.get_table_schema.return_value = None

            result = handler.get_schema(["some_table"])

        assert "Error fetching schema from database: db schema fetch failed" in result
