"""Conversation storage for the database chat application."""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from django.conf import settings

logger = logging.getLogger(__name__)


class ConversationStorage(ABC):
    """Abstract base class for conversation storage implementations."""

    @abstractmethod
    def save_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        """Save a message to the conversation history.

        Args:
            conversation_id: The unique identifier for the conversation
            message: The message to store, should contain 'role' and 'content' at minimum

        Returns:
            bool: True if the message was saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_conversation(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve the conversation history.

        Args:
            conversation_id: The unique identifier for the conversation
            limit: Optional limit on the number of messages to retrieve

        Returns:
            List of message dictionaries in chronological order
        """
        pass

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: The unique identifier for the conversation

        Returns:
            bool: True if the conversation was deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    def create_conversation(self) -> str:
        """Create a new conversation.

        Returns:
            str: The unique identifier for the new conversation
        """
        pass

    @abstractmethod
    def update_expiry(self, conversation_id: str, ttl_seconds: int) -> bool:
        """Update the expiration time for a conversation.

        Args:
            conversation_id: The unique identifier for the conversation
            ttl_seconds: Time to live in seconds

        Returns:
            bool: True if the expiration was updated successfully, False otherwise
        """
        pass

    @abstractmethod
    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists.

        Args:
            conversation_id: The unique identifier for the conversation
        """
        pass


class RedisConversationStorage(ConversationStorage):
    """Redis-based implementation of conversation storage."""

    def __init__(
        self, redis_url: Optional[str] = None, ttl_seconds: int = 60 * 60 * 24 * 7
    ):
        """Initialize Redis connection.

        Args:
            redis_url: Redis connection URL, defaults to settings.REDIS_URL
            ttl_seconds: Default TTL for conversation data
        """
        if not REDIS_AVAILABLE:
            logger.warning(
                "Redis package not available. RedisConversationStorage will not function. Install with 'pip install redis'"
            )
            self.redis_client = None
            return

        self.redis_url = redis_url or getattr(
            settings, "REDIS_URL", "redis://localhost:6379/0"
        )
        self.ttl_seconds = ttl_seconds

        try:
            self.redis_client = redis.Redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {self.redis_url}: {e}")
            logger.warning("Conversation history will not be persisted.")
            self.redis_client = None

    def _get_conversation_key(self, conversation_id: str) -> str:
        """Get the Redis key for a conversation."""
        return f"conversation:{conversation_id}"

    def save_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        """Save a message to the conversation history in Redis."""
        if not self.redis_client:
            logger.debug("Redis client not available, message not saved")
            return False

        try:
            if "timestamp" not in message:
                message["timestamp"] = datetime.now(timezone.utc).isoformat()

            message_json = json.dumps(message)
            conversation_key = self._get_conversation_key(conversation_id)

            self.redis_client.rpush(conversation_key, message_json)
            self.redis_client.expire(conversation_key, self.ttl_seconds)

            return True
        except Exception as e:
            logger.error(f"Error saving message to Redis: {e}")
            return False

    def get_conversation(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve the conversation history from Redis."""
        if not self.redis_client:
            logger.debug("Redis client not available, returning empty conversation")
            return []

        try:
            conversation_key = self._get_conversation_key(conversation_id)

            if not self.conversation_exists(conversation_id):
                return []

            messages_json = self.redis_client.lrange(conversation_key, 0, -1)
            messages = [json.loads(msg) for msg in messages_json]

            if limit is not None and limit > 0:
                messages = messages[-limit:]

            return messages
        except Exception as e:
            logger.error(f"Error retrieving conversation from Redis: {e}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from Redis."""
        if not self.redis_client:
            logger.debug("Redis client not available, cannot delete conversation")
            return False

        try:
            conversation_key = self._get_conversation_key(conversation_id)
            self.redis_client.delete(conversation_key)
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation from Redis: {e}")
            return False

    def create_conversation(self) -> str:
        """Create a new conversation in Redis."""
        conversation_id = str(uuid.uuid4())

        if self.redis_client:
            try:
                conversation_key = self._get_conversation_key(conversation_id)
                self.redis_client.set(conversation_key, "", ex=self.ttl_seconds)
                self.redis_client.delete(conversation_key)
            except Exception as e:
                logger.error(f"Error creating conversation in Redis: {e}")

        return conversation_id

    def update_expiry(
        self, conversation_id: str, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Update the expiration time for a conversation in Redis."""
        if not self.redis_client:
            logger.debug("Redis client not available, cannot update expiry")
            return False

        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds

        try:
            conversation_key = self._get_conversation_key(conversation_id)

            if self.conversation_exists(conversation_id):
                self.redis_client.expire(conversation_key, ttl)
                return True
            else:
                logger.debug(
                    f"Tried to update expiry for non-existent conversation: {conversation_id}"
                )
                return False

        except Exception as e:
            logger.error(f"Error updating conversation expiry in Redis: {e}")
            return False

    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists in Redis."""
        if not self.redis_client:
            logger.debug(
                "Redis client not available, cannot check conversation existence"
            )
            return False

        return self.redis_client.exists(self._get_conversation_key(conversation_id))


class InMemoryConversationStorage(ConversationStorage):
    """In-memory implementation of conversation storage.

    This is a fallback storage that doesn't persist between server restarts.
    Use only for development or when Redis is not available.
    """

    # Class-level storage - shared across instances
    _conversations = {}

    def __init__(self, ttl_seconds: int = 60 * 60 * 24):  # 1 day default
        """Initialize in-memory storage.

        Args:
            ttl_seconds: Default TTL for conversation data (not enforced automatically)
        """
        self.ttl_seconds = ttl_seconds
        logger.warning(
            "Using InMemoryConversationStorage - conversations will not persist between server restarts"
        )

    def save_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        """Save a message to the conversation history in memory."""
        try:
            if "timestamp" not in message:
                message["timestamp"] = datetime.now(timezone.utc).isoformat()

            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []

            self._conversations[conversation_id].append(message)

            return True
        except Exception as e:
            logger.error(f"Error saving message to memory: {e}")
            return False

    def get_conversation(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve the conversation history from memory."""
        try:
            if conversation_id not in self._conversations:
                return []

            messages = self._conversations[conversation_id]

            if limit is not None and limit > 0:
                messages = messages[-limit:]

            return messages
        except Exception as e:
            logger.error(f"Error retrieving conversation from memory: {e}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from memory."""
        try:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation from memory: {e}")
            return False

    def create_conversation(self) -> str:
        """Create a new conversation in memory."""
        conversation_id = str(uuid.uuid4())
        self._conversations[conversation_id] = []
        return conversation_id

    def update_expiry(
        self, conversation_id: str, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Update the expiration time for a conversation in memory.

        Note: This is a no-op for in-memory storage - expiry is not enforced.
        """
        return conversation_id in self._conversations

    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists in memory."""
        return conversation_id in self._conversations


# Factory function to get the configured storage implementation
@lru_cache(maxsize=1)
def get_conversation_storage() -> ConversationStorage:
    """Get the configured conversation storage implementation.

    This factory function examines settings to determine which
    storage implementation to use. Results are cached.

    Returns:
        An instance of ConversationStorage
    """
    storage_type = getattr(settings, "CONVERSATION_STORAGE_TYPE", "redis").lower()

    if storage_type == "redis":
        if not REDIS_AVAILABLE:
            logger.warning(
                "Redis package not installed, falling back to in-memory storage"
            )
            return InMemoryConversationStorage()

        redis_url = getattr(settings, "REDIS_URL", None)
        ttl_seconds = getattr(settings, "CONVERSATION_TTL_SECONDS", 60 * 60 * 24 * 7)

        storage = RedisConversationStorage(redis_url=redis_url, ttl_seconds=ttl_seconds)

        if not storage.redis_client:
            logger.warning(
                "Failed to connect to Redis, falling back to in-memory storage"
            )
            return InMemoryConversationStorage()

        return storage
    elif (
        storage_type == "memory"
        or storage_type == "in_memory"
        or storage_type == "inmemory"
    ):
        return InMemoryConversationStorage()
    else:
        logger.warning(
            f"Unknown storage type '{storage_type}', falling back to in-memory storage"
        )
        return InMemoryConversationStorage()


# == Vector Storage ==


class BaseVectorStorage(ABC):
    """Abstract base class for storing and retrieving message embeddings."""

    @abstractmethod
    def upsert_message(
        self, conversation_id: str, message: Dict[str, Any], vector: List[float]
    ):
        """Store a message along with its embedding vector."""
        pass

    @abstractmethod
    def search_relevant(
        self, conversation_id: str, query_vector: List[float], k: int = 3
    ) -> List[Dict[str, Any]]:
        """Find the top k messages most similar to the query vector."""
        pass

    @abstractmethod
    def delete_conversation_vectors(self, conversation_id: str):
        """Remove all vectors associated with a conversation."""
        pass


class InMemoryVectorStorage(BaseVectorStorage):
    """Simple in-memory vector storage using cosine similarity.

    Note: Requires `numpy` for calculations. Install with `pip install numpy`.
    This is suitable for development or small-scale use only.
    """

    _storage: Dict[str, List[Tuple[Dict[str, Any], List[float]]]] = {}
    _numpy_available = False

    def __init__(self):
        try:
            import numpy as np

            self._np = np
            self._numpy_available = True
        except ImportError:
            logger.warning(
                "NumPy not found. InMemoryVectorStorage similarity search will be basic. "
                "Install with 'pip install numpy'."
            )

    def upsert_message(
        self, conversation_id: str, message: Dict[str, Any], vector: List[float]
    ):
        if conversation_id not in self._storage:
            self._storage[conversation_id] = []
        self._storage[conversation_id].append((message, vector))

    def search_relevant(
        self, conversation_id: str, query_vector: List[float], k: int = 3
    ) -> List[Dict[str, Any]]:
        if not self._numpy_available or conversation_id not in self._storage:
            return []

        conv_history = self._storage[conversation_id]
        if not conv_history:
            return []

        # Calculate cosine similarity
        query_vec = self._np.array(query_vector)
        message_vecs = self._np.array([item[1] for item in conv_history])

        # Handle potential zero vectors
        query_norm = self._np.linalg.norm(query_vec)
        message_norms = self._np.linalg.norm(message_vecs, axis=1)

        if query_norm == 0:
            return []

        # Avoid division by zero for message vectors
        valid_indices = message_norms > 0
        if not self._np.any(valid_indices):
            return []

        similarities = self._np.dot(message_vecs[valid_indices], query_vec) / (
            message_norms[valid_indices] * query_norm
        )

        # Get indices of top k similarities
        valid_history = [
            conv_history[i] for i, valid in enumerate(valid_indices) if valid
        ]
        k = min(k, len(similarities))
        top_k_indices = self._np.argsort(similarities)[-k:][::-1]

        return [valid_history[i][0] for i in top_k_indices]

    def delete_conversation_vectors(self, conversation_id: str):
        if conversation_id in self._storage:
            del self._storage[conversation_id]


# Factory function for vector storage
@lru_cache(maxsize=1)
def get_vector_storage() -> BaseVectorStorage:
    from django.conf import settings

    from .components import import_from_dotted_path  # Reuse helper

    config = getattr(settings, "DB_CHAT", {}) or {}
    storage_path = config.get(
        "VECTOR_STORAGE_CLASS", "db_chat.storage.InMemoryVectorStorage"
    )
    try:
        storage_class = import_from_dotted_path(storage_path)
        instance = storage_class()
        logger.info(f"Using vector storage: {storage_class.__name__}")
        return instance
    except Exception as e:
        logger.exception(f"Failed to load vector storage '{storage_path}': {e}")
        logger.warning("Falling back to InMemoryVectorStorage for vectors.")
        return InMemoryVectorStorage()
