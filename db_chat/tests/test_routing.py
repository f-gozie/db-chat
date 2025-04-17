import pytest

from db_chat import routing


def test_websocket_urlpatterns():
    assert hasattr(routing, "websocket_urlpatterns")
    patterns = routing.websocket_urlpatterns
    assert isinstance(patterns, list)
    assert any("ws/db_chat/query/" in str(p.pattern) for p in patterns)
