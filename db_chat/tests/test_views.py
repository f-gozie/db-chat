import json
from unittest.mock import MagicMock, patch

import pytest
from django.test import Client

from db_chat import views


@pytest.mark.django_db
def test_model_registry_diagnostic():
    client = Client()
    resp = client.get("/model-registry-diagnostic/")
    assert resp.status_code == 200
    assert "initialized" in resp.json()


@pytest.mark.django_db
def test_chat_view_success():
    client = Client()
    with patch("db_chat.views.ChatService") as mock_service:
        mock_service.return_value.handle_query.return_value = {
            "reply": "ok",
            "sql_query": "SELECT 1;",
            "raw_result": "1",
            "conversation_id": "cid",
        }
        data = {"query": "show users"}
        resp = client.post(
            "/chat/", data=json.dumps(data), content_type="application/json"
        )
        assert resp.status_code == 200
        assert resp.json()["reply"] == "ok"


@pytest.mark.django_db
def test_chat_view_missing_query():
    client = Client()
    data = {}
    resp = client.post("/chat/", data=json.dumps(data), content_type="application/json")
    assert resp.status_code == 400
    assert "error" in resp.json()


@pytest.mark.django_db
def test_chat_view_invalid_json():
    client = Client()
    resp = client.post("/chat/", data="notjson", content_type="application/json")
    assert resp.status_code == 400
    assert "error" in resp.json()


@pytest.mark.django_db
def test_chat_view_internal_error():
    client = Client()
    with patch("db_chat.views.ChatService") as mock_service:
        mock_service.return_value.handle_query.side_effect = Exception("fail")
        data = {"query": "show users"}
        resp = client.post(
            "/chat/", data=json.dumps(data), content_type="application/json"
        )
        assert resp.status_code == 500
        assert "error" in resp.json()
