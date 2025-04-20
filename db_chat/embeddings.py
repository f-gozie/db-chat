"""Embedding model interfaces and default implementations."""
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    """Abstract base class for text embedding models."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Convert text into a vector embedding."""
        pass

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Return the dimension of the vectors produced by this model."""
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Embedding model using OpenAI's API."""

    DEFAULT_MODEL = "text-embedding-ada-002"
    DEFAULT_VECTOR_SIZE = 1536  # For text-embedding-ada-002

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            import openai
        except ImportError:
            logger.error("OpenAI package not found. Install with 'pip install openai'")
            raise

        from django.conf import settings

        self.client = openai.OpenAI(
            api_key=api_key or getattr(settings, "OPENAI_API_KEY", None)
        )
        if not self.client.api_key:
            raise ValueError(
                "OpenAI API key not found in settings (OPENAI_API_KEY) or arguments."
            )

        self.model = model or self.DEFAULT_MODEL
        # Try to determine vector size based on model, fallback to default
        # This is a basic example; a real implementation might inspect model details
        self._vector_size = self.DEFAULT_VECTOR_SIZE
        logger.info(f"Initialized OpenAIEmbeddingModel with model: {self.model}")

    def embed(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        except Exception as e:
            logger.exception(f"OpenAI embedding failed: {e}")
            return [0.0] * self.vector_size  # Fallback to zero vector

    @property
    def vector_size(self) -> int:
        return self._vector_size


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """Embedding model using Sentence Transformers library."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "Sentence Transformers package not found. "
                "Install with 'pip install sentence-transformers' (and potentially torch/tensorflow)."
            )
            raise

        from django.conf import settings

        config = getattr(settings, "DB_CHAT", {}) or {}
        resolved_model_name = model_name or config.get(
            "SENTENCE_TRANSFORMER_MODEL", self.DEFAULT_MODEL
        )

        try:
            self.model = SentenceTransformer(resolved_model_name)
            self._vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Initialized SentenceTransformerEmbeddingModel with model: {resolved_model_name} "
                f"(Vector size: {self._vector_size})"
            )
        except Exception as e:
            logger.exception(
                f"Failed to load Sentence Transformer model '{resolved_model_name}': {e}"
            )
            raise ValueError(
                f"Could not load Sentence Transformer model: {resolved_model_name}"
            ) from e

    def embed(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(text)
            # Ensure it's a list of floats
            return (
                embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(map(float, embedding))
            )
        except Exception as e:
            logger.exception(f"Sentence Transformer embedding failed: {e}")
            return [0.0] * self.vector_size  # Fallback

    @property
    def vector_size(self) -> int:
        return self._vector_size


class MockEmbeddingModel(BaseEmbeddingModel):
    """Placeholder embedding model that returns zero vectors.

    Use this only for testing or when no real embedding model is configured.
    Vector size is arbitrarily set to 10.
    """

    def embed(self, text: str) -> List[float]:
        # Return a zero vector of the expected size
        return [0.0] * self.vector_size

    @property
    def vector_size(self) -> int:
        return 10


# Factory function for embedding model
@lru_cache(maxsize=1)
def get_embedding_model() -> BaseEmbeddingModel:
    from django.conf import settings

    from .components import import_from_dotted_path

    config = getattr(settings, "DB_CHAT", {}) or {}
    model_path = config.get(
        "EMBEDDING_MODEL_CLASS", "db_chat.embeddings.MockEmbeddingModel"
    )
    try:
        model_class = import_from_dotted_path(model_path)
        instance = model_class()
        logger.info(f"Using embedding model: {model_class.__name__}")
        return instance
    except ImportError as ie:
        logger.warning(
            f"Import error loading '{model_path}': {ie}. Dependency likely missing."
        )
        logger.warning("Falling back to MockEmbeddingModel.")
        return MockEmbeddingModel()
    except Exception as e:  # Catch other init errors (e.g., API key missing)
        logger.exception(f"Failed to load embedding model '{model_path}': {e}")
        logger.warning("Falling back to MockEmbeddingModel.")
        return MockEmbeddingModel()
