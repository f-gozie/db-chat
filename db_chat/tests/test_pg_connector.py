from unittest.mock import MagicMock, patch

import pytest

from db_chat.connectors.postgres.pg_connector import PgConnector


@patch("db_chat.connectors.postgres.pg_connector.psycopg2.connect")
def test_connect_success(mock_connect):
    mock_conn = MagicMock()
    mock_conn.closed = False
    mock_connect.return_value = mock_conn
    connector = PgConnector("fake_conn_str")
    result = connector.connect()
    assert result is True
    assert connector.conn == mock_conn


@patch("db_chat.connectors.postgres.pg_connector.psycopg2.connect")
def test_connect_failure(mock_connect):
    mock_connect.side_effect = Exception("fail")
    connector = PgConnector("fake_conn_str")
    result = connector.connect()
    assert result is False
    assert connector.conn is None


@patch("db_chat.connectors.postgres.pg_connector.psycopg2.connect")
def test_connect_already_connected(mock_connect):
    connector = PgConnector("fake_conn_str")
    connector.conn = MagicMock()
    connector.conn.closed = False
    result = connector.connect()
    assert result is True
    mock_connect.assert_not_called()


def test_disconnect_success():
    connector = PgConnector("fake_conn_str")
    mock_conn = MagicMock()
    mock_conn.closed = False
    connector.conn = mock_conn
    connector.disconnect()
    mock_conn.close.assert_called_once()
    assert connector.conn is None


def test_disconnect_no_connection():
    connector = PgConnector("fake_conn_str")
    connector.conn = None
    # Should not raise
    connector.disconnect()


def test_disconnect_error():
    connector = PgConnector("fake_conn_str")
    bad_conn = MagicMock()
    bad_conn.close.side_effect = Exception("fail")
    connector.conn = bad_conn
    # Should not raise
    connector.disconnect()


def test_get_connection_reconnect():
    connector = PgConnector("fake_conn_str")
    connector.conn = MagicMock()
    connector.conn.closed = True
    with patch.object(connector, "connect", return_value=True) as mock_connect:
        connector.conn = MagicMock()
        connector.conn.closed = True
        result = connector.get_connection()
        mock_connect.assert_called_once()


def test_get_connection_fail_reconnect():
    connector = PgConnector("fake_conn_str")
    connector.conn = None
    with patch.object(connector, "connect", return_value=False):
        result = connector.get_connection()
        assert result is None
