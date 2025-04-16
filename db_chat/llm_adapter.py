"""Adapters for various LLM providers."""

import logging
from abc import ABC, abstractmethod
from django.conf import settings
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    @abstractmethod
    def generate_text(self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        """Generate text from the LLM based on system prompt and messages.
        
        Args:
            system_prompt: System prompt for the LLM
            messages: List of message dictionaries with role and content
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        pass
    
    @classmethod
    def get_adapter(cls) -> 'LLMAdapter':
        """Factory method to get the appropriate LLM adapter based on settings.
        
        Returns:
            LLMAdapter: Instance of the appropriate LLM adapter
        """
        provider = getattr(settings, 'LLM_PROVIDER', 'anthropic').lower()
        
        if provider == 'anthropic':
            return AnthropicAdapter()
        elif provider == 'openai':
            return OpenAIAdapter()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude models."""
    
    def __init__(self):
        """Initialize the Anthropic adapter."""
        from anthropic import Anthropic
        
        self.api_key = settings.ANTHROPIC_API_KEY
        self.model = getattr(settings, 'ANTHROPIC_MODEL', 'claude-3-5-haiku-latest')
        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"Initialized Anthropic adapter with model: {self.model}")
    
    def generate_text(self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        """Generate text using Anthropic's Claude.
        
        Args:
            system_prompt: System prompt for Claude
            messages: List of message dictionaries with role and content
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                max_tokens=max_tokens
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            raise


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI models."""
    
    def __init__(self):
        """Initialize the OpenAI adapter."""
        try:
            import openai
            
            self.api_key = settings.OPENAI_API_KEY
            self.model = getattr(settings, 'OPENAI_MODEL', 'gpt-4o')
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI adapter with model: {self.model}")
        except ImportError:
            logger.error("OpenAI package not installed. Please install with: pip install openai")
            raise
    
    def generate_text(self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        """Generate text using OpenAI's models.
        
        Args:
            system_prompt: System prompt for the model
            messages: List of message dictionaries with role and content
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        try:
            # Add system message to messages list (OpenAI format)
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise 