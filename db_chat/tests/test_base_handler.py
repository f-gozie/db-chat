import pytest

from db_chat.connectors.base import BaseConnector
from db_chat.connectors.handlers.base import BaseHandler


class DummyConnector(BaseConnector):
    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_connection(self):
        pass


class DummyHandler(BaseHandler):
    def execute_query(self, sql_query, params=None):
        pass

    def get_schema(self, tables=None):
        pass


def test_format_results_empty():
    h = DummyHandler(DummyConnector())
    assert h._format_results([]) == "No results found."


def test_format_results_non_dict(caplog):
    h = DummyHandler(DummyConnector())
    out = h._format_results([1, 2, 3])
    assert out == "[1, 2, 3]"
    assert any(
        "Attempted to format non-dict results" in m for m in caplog.text.splitlines()
    )


def test_format_results_no_columns(caplog):
    h = DummyHandler(DummyConnector())
    out = h._format_results([{}])
    assert out == "Query returned rows with no columns."
    assert any(
        "Attempted to format results with no columns" in m
        for m in caplog.text.splitlines()
    )


def test_format_results_normal():
    h = DummyHandler(DummyConnector())
    results = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    out = h._format_results(results)
    assert "| id | name |" in out
    assert "| 1 | Alice |" in out
    assert "| 2 | Bob |" in out
    assert "2 row(s) returned." in out
