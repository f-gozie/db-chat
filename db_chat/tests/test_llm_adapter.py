import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from db_chat.llm_adapter import AnthropicAdapter, LLMAdapter, OpenAIAdapter


@pytest.fixture(autouse=True)
def mock_langchain_imports(monkeypatch):
    mock_chat_anthropic = MagicMock(name="MockChatAnthropic")
    mock_chat_openai = MagicMock(name="MockChatOpenAI")
    monkeypatch.setattr("db_chat.llm_adapter.ChatAnthropic", mock_chat_anthropic)
    monkeypatch.setattr("db_chat.llm_adapter.ChatOpenAI", mock_chat_openai)
    yield {"anthropic": mock_chat_anthropic, "openai": mock_chat_openai}


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    class DummySettings:
        LLM_PROVIDER = "anthropic"
        ANTHROPIC_API_KEY = "fake-key"
        ANTHROPIC_MODEL = "claude-test-model"
        OPENAI_API_KEY = "fake-key"
        OPENAI_MODEL = "gpt-test-model"

    monkeypatch.setattr("db_chat.llm_adapter.settings", DummySettings)
    yield


class TestLLMAdapter:
    def test_get_adapter_anthropic(self, mock_langchain_imports):
        mock_langchain_imports["anthropic"].reset_mock()
        adapter = LLMAdapter.get_adapter()
        assert isinstance(adapter, AnthropicAdapter)
        mock_langchain_imports["anthropic"].assert_called_once_with(
            model="claude-test-model", anthropic_api_key="fake-key"
        )

    def test_get_adapter_openai(self, monkeypatch, mock_langchain_imports):
        monkeypatch.setattr("db_chat.llm_adapter.settings.LLM_PROVIDER", "openai")
        mock_langchain_imports["openai"].reset_mock()
        adapter = LLMAdapter.get_adapter()
        assert isinstance(adapter, OpenAIAdapter)
        mock_langchain_imports["openai"].assert_called_once_with(
            model="gpt-test-model", openai_api_key="fake-key"
        )

    def test_get_adapter_invalid(self, monkeypatch):
        monkeypatch.setattr("db_chat.llm_adapter.settings.LLM_PROVIDER", "invalid")
        with pytest.raises(ValueError):
            LLMAdapter.get_adapter()

    def test_anthropic_generate_text_success(self, mock_langchain_imports):
        adapter = LLMAdapter.get_adapter()
        mock_client = adapter.client
        mock_client.invoke.return_value = AIMessage(content="langchain result")

        result = adapter.generate_text("sys", [{"role": "user", "content": "hi"}])
        assert result == "langchain result"
        mock_client.invoke.assert_called_once()

    def test_openai_generate_text_success(self, monkeypatch, mock_langchain_imports):
        monkeypatch.setattr("db_chat.llm_adapter.settings.LLM_PROVIDER", "openai")
        adapter = LLMAdapter.get_adapter()
        mock_client = adapter.client
        mock_client.invoke.return_value = AIMessage(content="langchain result")

        result = adapter.generate_text("sys", [{"role": "user", "content": "hi"}])
        assert result == "langchain result"
        mock_client.invoke.assert_called_once()

    def test_openai_init_import_error(self, monkeypatch, mock_langchain_imports):
        monkeypatch.setattr("db_chat.llm_adapter.settings.LLM_PROVIDER", "openai")

        mock_langchain_imports["openai"].side_effect = ImportError(
            "cannot import name 'OpenAI' from partially initialized module 'openai'"
        )

        with pytest.raises(ImportError):
            LLMAdapter.get_adapter()

    def test_adapter_message_conversion(self):
        adapter = AnthropicAdapter()
        system_prompt = "System instructions"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {
                "role": "system",
                "content": "This should be ignored by convert",
            },  # Example if filtering needed
        ]
        lc_messages = adapter._convert_messages(system_prompt, messages)

        assert len(lc_messages) == 4
        assert lc_messages[0].type == "system"
        assert lc_messages[0].content == "System instructions"
        assert lc_messages[1].type == "human"
        assert lc_messages[1].content == "Hello"
        assert lc_messages[2].type == "ai"
        assert lc_messages[2].content == "Hi there!"
        assert lc_messages[3].type == "human"
        assert lc_messages[3].content == "How are you?"
