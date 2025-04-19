import asyncio

import pytest

import db_chat.prompts as prompts
from db_chat import error_handlers
from db_chat.constants import DatabaseDialects


def test_register_and_get_handler():
    class Dummy(error_handlers.BaseErrorHandler):
        async def stream(self, *a, **k):
            yield {"type": "llm_token", "token": "dummy", "conversation_id": "cid"}

    error_handlers.register_error_handler("dummy_type", Dummy)
    handler = error_handlers.get_handler("dummy_type")
    assert isinstance(handler, Dummy)
    # Fallback to DefaultErrorHandler
    fallback = error_handlers.get_handler("notype")
    assert isinstance(fallback, error_handlers.DefaultErrorHandler)


import pytest


@pytest.mark.asyncio
async def test_default_error_handler_stream():
    from db_chat import error_handlers

    handler = error_handlers.DefaultErrorHandler()

    class DummyService:
        async def save_message(self, *a, **k):
            return True

    service = DummyService()
    gen = handler.stream(service, "q", "sql", "err", "schema", "cid")
    results = [msg async for msg in gen]
    assert any(m["type"] == "llm_token" for m in results)
    assert any(m["type"] == "llm_stream_end" for m in results)


@pytest.mark.asyncio
async def test_llm_error_handler_stream_exception():
    from db_chat import error_handlers

    class DummyService:
        def _build_messages_for_llm(self, cid):
            return []

        class llm_adapter:
            @staticmethod
            async def stream_text(system_prompt, messages):
                raise Exception("llm fail")

        async def save_message(self, *a, **k):
            return True

    handler = error_handlers.LLMErrorHandler()
    service = DummyService()
    gen = handler.stream(service, "q", "sql", "err", "schema", "cid")
    results = [msg async for msg in gen]
    assert any(m["type"] == "error" for m in results)
    assert any("error" in m["type"] or m.get("message") for m in results)


@pytest.mark.asyncio
async def test_non_sql_handler_stream_normal():
    handler = error_handlers.NonSQLHandler()

    class DummyService:
        def _build_messages_for_llm(self, cid):
            return []

        class llm_adapter:
            @staticmethod
            async def stream_text(system_prompt, messages):
                for t in ["Try asking about users."]:
                    yield t

        async def save_message(self, *a, **k):
            return True

    service = DummyService()
    gen = handler.stream(service, "q", "sql", "err", "schema", "cid")
    results = [msg async for msg in gen]
    assert any(m["type"] == "llm_token" for m in results)
    assert any(m["type"] == "llm_stream_end" for m in results)


@pytest.mark.asyncio
async def test_non_sql_handler_stream_exception():
    handler = error_handlers.NonSQLHandler()

    class DummyService:
        def _build_messages_for_llm(self, cid):
            return []

        class llm_adapter:
            @staticmethod
            async def stream_text(system_prompt, messages):
                raise Exception("fail")

        async def save_message(self, *a, **k):
            return True

    service = DummyService()
    gen = handler.stream(service, "q", "sql", "err", "schema", "cid")
    results = [msg async for msg in gen]
    assert any(m["type"] == "llm_token" for m in results)
    assert any(m["type"] == "llm_stream_end" for m in results)
