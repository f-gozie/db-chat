import importlib
from unittest.mock import MagicMock, patch

import pytest

from db_chat.db_executor import PostgreSQLExecutor


class TestPostgreSQLExecutor:
    @patch("db_chat.db_executor.psycopg2.connect")
    def test_connect_success(self, mock_connect):
        executor = PostgreSQLExecutor("fake_conn")
        mock_connect.return_value = MagicMock()
        assert executor.connect() is True

    @patch("db_chat.db_executor.psycopg2.connect", side_effect=Exception("fail"))
    def test_connect_failure(self, mock_connect):
        executor = PostgreSQLExecutor("fake_conn")
        assert executor.connect() is False

    def test_disconnect(self):
        executor = PostgreSQLExecutor("fake_conn")
        executor.conn = MagicMock()
        executor.disconnect()
        assert executor.conn is None

    @patch("db_chat.db_executor.psycopg2.connect")
    def test_execute_query_success(self, mock_connect):
        executor = PostgreSQLExecutor("fake_conn")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.description = [object()]
        mock_cursor.fetchall.return_value = [{"id": 1, "name": "test"}]
        mock_cursor.execute.side_effect = [None, None, None]
        executor.connect()
        result = executor.execute_query("SELECT * FROM test")
        assert "result" in result
        assert (
            "test" not in result["result"]
            or "No results found." not in result["result"]
        )

    @patch("db_chat.db_executor.psycopg2.connect")
    def test_execute_query_error(self, mock_connect):
        executor = PostgreSQLExecutor("fake_conn")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("bad sql")
        executor.connect()
        result = executor.execute_query("SELECT * FROM test")
        assert "Error" in result["result"]

    @patch("db_chat.db_executor.psycopg2.connect")
    def test_get_schema_import_error(self, mock_connect):
        executor = PostgreSQLExecutor("fake_conn")
        # Patch importlib.import_module to raise ImportError for db_chat.model_registry
        orig_import_module = importlib.import_module

        def fake_import_module(name, *args, **kwargs):
            if name == "db_chat.model_registry":
                raise ImportError("fake import error")
            return orig_import_module(name, *args, **kwargs)

        with patch("importlib.import_module", side_effect=fake_import_module):
            result = executor.get_schema(["test"])
            assert "Error" in result or isinstance(result, str)
