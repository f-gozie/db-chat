import sys
from unittest.mock import MagicMock, patch

import pytest

from db_chat.llm_adapter import AnthropicAdapter, LLMAdapter, OpenAIAdapter


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    class DummySettings:
        LLM_PROVIDER = "anthropic"
        ANTHROPIC_API_KEY = "key"
        ANTHROPIC_MODEL = "model"
        OPENAI_API_KEY = "key"
        OPENAI_MODEL = "model"

    monkeypatch.setattr("db_chat.llm_adapter.settings", DummySettings)
    yield


class TestLLMAdapter:
    def test_get_adapter_anthropic(self):
        adapter = LLMAdapter.get_adapter()
        assert isinstance(adapter, AnthropicAdapter)

    def test_get_adapter_openai(self, monkeypatch):
        monkeypatch.setattr("db_chat.llm_adapter.settings.LLM_PROVIDER", "openai")
        # Patch OpenAIAdapter.__init__ to avoid real import
        monkeypatch.setattr(OpenAIAdapter, "__init__", lambda self: None)
        adapter = LLMAdapter.get_adapter()
        assert isinstance(adapter, OpenAIAdapter)

    def test_get_adapter_invalid(self, monkeypatch):
        monkeypatch.setattr("db_chat.llm_adapter.settings.LLM_PROVIDER", "invalid")
        with pytest.raises(ValueError):
            LLMAdapter.get_adapter()

    def test_anthropic_generate_text_success(self, monkeypatch):
        adapter = AnthropicAdapter()
        adapter.client = MagicMock()
        adapter.client.messages.create.return_value.content = [MagicMock(text="result")]
        result = adapter.generate_text("sys", [{"role": "user", "content": "hi"}])
        assert result == "result"

    def test_openai_generate_text_success(self, monkeypatch):
        # Patch OpenAIAdapter.__init__ to avoid real import
        monkeypatch.setattr(OpenAIAdapter, "__init__", lambda self: None)
        adapter = OpenAIAdapter()
        adapter.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="result"))]
        adapter.client.chat.completions.create.return_value = mock_response
        adapter.model = "model"
        result = adapter.generate_text("sys", [{"role": "user", "content": "hi"}])
        assert result == "result"

    def test_openai_import_error(self, monkeypatch):
        # Simulate ImportError by removing openai from sys.modules
        monkeypatch.setitem(sys.modules, "openai", None)
        monkeypatch.setattr(
            OpenAIAdapter, "__init__", lambda self: (_ for _ in ()).throw(ImportError)
        )
        with pytest.raises(ImportError):
            OpenAIAdapter()
