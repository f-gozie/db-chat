"""Error handling strategy implementations for scalable, extensible explanations.

Each error type is mapped to a handler that knows how to stream a
user-friendly explanation back to the websocket.  Developers can register
custom handlers for new error types without modifying core logic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, Dict, Optional, Type

from . import prompts

StatusCb = Optional[Callable[[str], asyncio.Future]]


class BaseErrorHandler(ABC):
    """Abstract base class for error-handling streaming strategies."""

    @abstractmethod
    async def stream(
        self,
        chat_service: "ChatService",
        user_query: str,
        executed_sql: str,
        error_message: str,
        schema: str,
        conversation_id: str,
        status_callback: StatusCb = None,
    ) -> AsyncGenerator[Dict, None]:
        """Yield websocket chunks explaining the error in real time."""
        raise NotImplementedError


_HANDLER_REGISTRY: Dict[str, Type[BaseErrorHandler]] = {}


def register_error_handler(error_type: str, handler_cls: Type[BaseErrorHandler]):
    """Allow third-party code to plug in custom handlers."""
    _HANDLER_REGISTRY[error_type] = handler_cls


def get_handler(error_type: str) -> BaseErrorHandler:
    handler_cls = _HANDLER_REGISTRY.get(error_type, DefaultErrorHandler)
    return handler_cls()


class LLMErrorHandler(BaseErrorHandler):
    """Uses the LLM to explain SQL-related errors (validation, execution, etc.)."""

    async def stream(
        self,
        chat_service,
        user_query: str,
        executed_sql: str,
        error_message: str,
        schema: str,
        conversation_id: str,
        status_callback: StatusCb = None,
    ):
        if status_callback:
            await status_callback("Generating friendly error explanation via LLM…")

        if executed_sql:
            user_prompt = prompts.get_error_user_prompt(executed_sql, error_message)
            system_prompt = prompts.get_error_system_prompt()
        else:
            user_prompt = prompts.get_user_friendly_error_prompt(
                user_query, "generic_error", error_message
            )
            system_prompt = prompts.get_error_system_prompt()

        messages_for_llm = chat_service._build_messages_for_llm(conversation_id) + [
            {"role": "user", "content": user_prompt}
        ]
        full = ""
        try:
            async for token in chat_service.llm_adapter.stream_text(
                system_prompt, messages_for_llm
            ):
                full += token
                yield {
                    "type": "llm_token",
                    "token": token,
                    "conversation_id": conversation_id,
                }
            yield {
                "type": "llm_stream_end",
                "message": full,
                "conversation_id": conversation_id,
            }
            chat_service.save_message(conversation_id, "assistant", full)
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Failed to generate error explanation: {e}",
                "conversation_id": conversation_id,
            }


class NonSQLHandler(BaseErrorHandler):
    """Handles ambiguous questions that don't translate to SQL (cannot_answer)."""

    async def stream(
        self,
        chat_service,
        user_query: str,
        executed_sql: str,
        error_message: str,
        schema: str,
        conversation_id: str,
        status_callback: StatusCb = None,
    ):
        static_reply = (
            "I'm your database assistant and can answer questions that involve data in your database. "
            "Your last question was a bit too broad for a database query. "
            "Try asking something like:\n"
            "• 'How many users signed up last week?'\n"
            "• 'List projects completed in 2023.'"
        )

        if status_callback:
            await status_callback("Providing suggestions for a more specific question…")

        try:
            system_prompt = "You are a helpful database assistant. Suggest how to rephrase the question for a data query."
            user_prompt = (
                f"The user asked: '{user_query}'.\n"
                "Explain briefly why that can't be answered with a SQL query and give 2‑3 example questions that *can* be answered."
            )
            messages = chat_service._build_messages_for_llm(conversation_id) + [
                {"role": "user", "content": user_prompt}
            ]
            full = ""
            async for tok in chat_service.llm_adapter.stream_text(
                system_prompt, messages
            ):
                full += tok
                yield {
                    "type": "llm_token",
                    "token": tok,
                    "conversation_id": conversation_id,
                }
            yield {
                "type": "llm_stream_end",
                "message": full,
                "conversation_id": conversation_id,
            }
            chat_service.save_message(conversation_id, "assistant", full)
        except Exception:
            yield {
                "type": "llm_token",
                "token": static_reply,
                "conversation_id": conversation_id,
            }
            yield {
                "type": "llm_stream_end",
                "message": static_reply,
                "conversation_id": conversation_id,
            }
            chat_service.save_message(conversation_id, "assistant", static_reply)


class DefaultErrorHandler(BaseErrorHandler):
    """Fallback when no specific handler is registered."""

    async def stream(
        self,
        chat_service,
        user_query: str,
        executed_sql: str,
        error_message: str,
        schema: str,
        conversation_id: str,
        status_callback: StatusCb = None,
    ):
        msg = "Sorry, something went wrong while processing your request. Please try again later."
        if status_callback:
            await status_callback("Returning generic error message…")
        yield {"type": "llm_token", "token": msg, "conversation_id": conversation_id}
        yield {
            "type": "llm_stream_end",
            "message": msg,
            "conversation_id": conversation_id,
        }
        chat_service.save_message(conversation_id, "assistant", msg)


# Register built‑in handlers
for _err in [
    "invalid_structure",
    "security_violation",
    "trailing_comma",
    "schema_error",
    "generation_execution_exception",
    "internal_processing_error",
    "sql_execution_error",
]:
    register_error_handler(_err, LLMErrorHandler)

register_error_handler("cannot_answer", NonSQLHandler)
