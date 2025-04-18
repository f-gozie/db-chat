"""Adapters for various LLM providers."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List

from django.conf import settings
from langchain_anthropic import ChatAnthropic
from langchain_core import messages
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    @abstractmethod
    def generate_text(
        self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1024
    ) -> str:
        """Generate text from the LLM based on system prompt and messages.

        Args:
            system_prompt: System prompt for the LLM
            messages: List of message dictionaries with role and content
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        pass

    @abstractmethod
    async def stream_text(
        self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1024
    ):
        """Async generator that yields text chunks from the LLM as they become available."""
        pass

    @classmethod
    def get_adapter(cls) -> "LLMAdapter":
        """Factory method to get the appropriate LLM adapter based on settings.

        Returns:
            LLMAdapter: Instance of the appropriate LLM adapter
        """
        provider = getattr(settings, "LLM_PROVIDER", "anthropic").lower()

        if provider == "anthropic":
            return AnthropicAdapter()
        elif provider == "openai":
            return OpenAIAdapter()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude models using LangChain."""

    def __init__(self):
        """Initialize the Anthropic LangChain adapter."""
        self.api_key = settings.ANTHROPIC_API_KEY
        self.model_name = getattr(
            settings, "ANTHROPIC_MODEL", "claude-3-5-haiku-latest"
        )
        self.client = ChatAnthropic(
            model=self.model_name,
            anthropic_api_key=self.api_key,
        )
        logger.info(
            f"Initialized LangChain Anthropic adapter with model: {self.model_name}"
        )

    def _convert_messages(
        self, system_prompt: str, message_dicts: List[Dict[str, str]]
    ) -> List[messages.BaseMessage]:
        """Converts messages to LangChain format, including system prompt."""
        lc_messages: List[messages.BaseMessage] = [
            messages.SystemMessage(content=system_prompt)
        ]
        for msg in message_dicts:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(messages.HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(messages.AIMessage(content=content))
            # Langchain typically doesn't pass 'system' messages mid-conversation
            # elif role == "system":
            #     # Handle potential system messages if needed, though usually just one at the start
            #     lc_messages.append(SystemMessage(content=content))
        return lc_messages

    def generate_text(
        self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1024
    ) -> str:
        """Generate text using LangChain's ChatAnthropic.

        Args:
            system_prompt: System prompt for Claude
            messages: List of message dictionaries with role and content
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        lc_messages = self._convert_messages(system_prompt, messages)
        try:
            response = self.client.invoke(lc_messages, max_tokens=max_tokens)
            if isinstance(response.content, str):
                return response.content.strip()
            else:
                logger.warning(
                    f"Received non-string content from Anthropic: {type(response.content)}"
                )
                return str(response.content).strip()
        except Exception as e:
            logger.error(f"Error generating text with LangChain Anthropic: {e}")
            raise

    async def stream_text(
        self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1024
    ):
        """Async generator that yields text chunks from Anthropic via LangChain."""
        lc_messages = self._convert_messages(system_prompt, messages)
        try:
            async for chunk in self.client.astream(lc_messages, max_tokens=max_tokens):
                if hasattr(chunk, "content"):
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming text with LangChain Anthropic: {e}")
            raise


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI models using LangChain."""

    def __init__(self):
        """Initialize the OpenAI LangChain adapter."""
        try:
            self.api_key = settings.OPENAI_API_KEY
            self.model_name = getattr(settings, "OPENAI_MODEL", "gpt-4o")
            self.client = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
            )
            logger.info(
                f"Initialized LangChain OpenAI adapter with model: {self.model_name}"
            )
        except ImportError:
            logger.error(
                "OpenAI or LangChain OpenAI packages not installed. Please install with: pip install langchain-openai openai"
            )
            raise
        except Exception as e:
            logger.error(f"Error initializing LangChain OpenAI Adapter: {e}")
            raise

    def _convert_messages(
        self, system_prompt: str, message_dicts: List[Dict[str, str]]
    ) -> List[messages.BaseMessage]:
        """Converts messages to LangChain format, including system prompt."""
        lc_messages: List[messages.BaseMessage] = [
            messages.SystemMessage(content=system_prompt)
        ]
        for msg in message_dicts:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(messages.HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(messages.AIMessage(content=content))
        return lc_messages

    def generate_text(
        self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1024
    ) -> str:
        """Generate text using LangChain's ChatOpenAI.

        Args:
            system_prompt: System prompt for the model
            messages: List of message dictionaries with role and content
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        lc_messages = self._convert_messages(system_prompt, messages)
        try:
            response = self.client.invoke(lc_messages, max_tokens=max_tokens)
            if isinstance(response.content, str):
                return response.content.strip()
            else:
                logger.warning(
                    f"Received non-string content from OpenAI: {type(response.content)}"
                )
                return str(response.content).strip()
        except Exception as e:
            logger.error(f"Error generating text with LangChain OpenAI: {e}")
            raise

    async def stream_text(
        self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1024
    ):
        """Async generator that yields text chunks from OpenAI via LangChain."""
        lc_messages = self._convert_messages(system_prompt, messages)
        try:
            async for chunk in self.client.astream(lc_messages, max_tokens=max_tokens):
                if hasattr(chunk, "content"):
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming text with LangChain OpenAI: {e}")
            raise
