import pytest

from db_chat.connectors.base import BaseConnector


class DummyConnector(BaseConnector):
    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_connection(self):
        pass


def test_format_results_empty():
    c = DummyConnector()
    assert c._format_results([]) == "No results found."


def test_format_results_no_columns():
    c = DummyConnector()
    assert c._format_results([{}]) == "Query returned rows with no columns."


def test_format_results_normal():
    c = DummyConnector()
    results = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    out = c._format_results(results)
    assert "| id | name |" in out
    assert "| 1 | Alice |" in out
    assert "| 2 | Bob |" in out
    assert "2 row(s) returned." in out
