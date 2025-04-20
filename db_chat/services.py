"""Core service for the database chat application."""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from asgiref.sync import sync_to_async
from django.conf import settings

from . import prompts
from .connectors.postgres.pg_connector import PgConnector
from .connectors.postgres.pg_handler import PgHandler
from .constants import DatabaseDialects
from .embeddings import BaseEmbeddingModel, get_embedding_model
from .error_handlers import get_handler
from .llm_adapter import LLMAdapter
from .model_registry import get_registry
from .storage import (
    BaseVectorStorage,
    ConversationStorage,
    get_conversation_storage,
    get_vector_storage,
)
from .utils import clean_sql_query, is_safe_sql, is_valid_sql_structure

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


# --- Token Counting Helper ---


def _estimate_token_count(
    messages: List[Dict[str, str]], model_name: str = "gpt-4"
) -> int:
    """Estimate token count for a list of messages using tiktoken."""
    if not TIKTOKEN_AVAILABLE:
        # Fallback: estimate based on character count (very rough)
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return total_chars // 4  # Rough approximation: 1 token ~= 4 chars

    try:
        # Attempt to get encoding for the specific model, fallback to cl100k_base
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            # Standard token counting logic for chat models
            num_tokens += (
                4  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":  # If there's a name, the role is omitted
                    num_tokens -= 1  # Role is always required and always 1 token
        num_tokens += 2  # Every reply is primed with <|start|>assistant
        return num_tokens
    except Exception as e:
        logger.warning(
            f"Tiktoken estimation failed for model {model_name}: {e}. Falling back to char count."
        )
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return total_chars // 4


class ChatService:
    """Orchestrates the interaction between user queries, the LLM, the database,
    and conversation history storage.

    This service acts as the central hub for the chat application, managing the
    flow of information and coordinating the different components involved in
    processing a user's request related to database information.

    Key Responsibilities:
        - Initializing and holding instances of the database connector/handler,
          LLM adapter, model registry, and conversation storage.
        - Managing conversation lifecycles (creation, context retrieval, expiry).
        - Translating natural language user queries into executable SQL queries
          using the LLM, guided by database schema and conversation history.
        - Validating generated SQL for safety and basic structural integrity.
        - Executing validated SQL queries against the database via the handler.
        - Interpreting potentially complex SQL query results into user-friendly,
          natural language responses using the LLM.
        - Handling various errors gracefully (e.g., schema fetching issues,
          SQL generation failures, execution errors, interpretation problems)
          and providing informative feedback to the user.

    Attributes:
        db_connector (PgConnector): Instance for managing the raw database connection.
        db_handler (PgHandler): Instance for executing queries and schema operations.
        llm_adapter (LLMAdapter): Instance for interacting with the Large Language Model.
        conversation_storage (ConversationStorage): Instance for storing/retrieving chat history.
        allowed_tables (List[str]): List of database table names the service is permitted
            to query, derived from the model registry.
        context_limit (int): The maximum number of past messages to include in the
            context provided to the LLM.
        db_dialect (str): The database dialect (e.g., "PostgreSQL").
        summarization_enabled (bool): Whether summarization is enabled.
        summarization_trigger (int): The trigger for summarization.
        vector_storage (BaseVectorStorage): Instance for storing vectors.
        embedding_model (BaseEmbeddingModel): Instance for generating embeddings.
        rag_enabled (bool): Whether RAG is enabled.
        rag_k (int): The number of relevant messages to retrieve.
        rag_m_recent (int): The number of recent messages to include.
    """

    db_connector: PgConnector
    db_handler: PgHandler
    llm_adapter: LLMAdapter
    conversation_storage: ConversationStorage
    allowed_tables: List[str]
    context_limit: int
    db_dialect: str
    summarization_enabled: bool
    summarization_trigger: int
    vector_storage: BaseVectorStorage
    embedding_model: BaseEmbeddingModel
    rag_enabled: bool
    rag_k: int
    rag_m_recent: int

    def __init__(self):
        """Initializes the ChatService by setting up all its dependent components.

        Calls private helper methods to initialize the database connection,
        model registry, LLM adapter, and conversation storage based on
        Django settings.
        """
        self._initialize_db_components()
        self._initialize_model_registry()
        self._initialize_llm_adapter()
        self._initialize_conversation_storage()
        # Config flags
        cfg = getattr(settings, "DB_CHAT", {}) or {}
        self.summarization_enabled = cfg.get("SUMMARIZATION_ENABLED", True)
        self.summarization_trigger = cfg.get("SUMMARIZATION_TRIGGER", 12)
        self.rag_enabled = cfg.get("RAG_ENABLED", False)
        self.rag_k = cfg.get("RAG_K", 3)
        self.rag_m_recent = cfg.get("RAG_M_RECENT", 2)

        # Initialize RAG components if enabled
        if self.rag_enabled:
            self.vector_storage = get_vector_storage()
            self.embedding_model = get_embedding_model()
            logger.info(
                "RAG enabled. Using Vector Store: %s, Embedding Model: %s",
                self.vector_storage.__class__.__name__,
                self.embedding_model.__class__.__name__,
            )
        else:
            self.vector_storage = None
            self.embedding_model = None

        logger.info("ChatService initialized successfully.")

    def _initialize_db_components(self):
        """Initializes the `PgConnector` and `PgHandler`.

        Reads database connection details (host, port, name, user, password)
        from the Django settings (`settings.DATABASES['default']`) and uses
        them to instantiate the PostgreSQL connector and handler.
        Logs the connection string pattern being used (without credentials).

        Raises:
            KeyError: If the 'default' database configuration is missing in settings.
            Exception: If connector or handler instantiation fails.
        """
        try:
            db_config = settings.DATABASES["default"]
            db_host = db_config.get("HOST", "postgres")
            db_port = db_config.get("PORT", "5432")
            db_name = db_config.get("NAME", "db_chat")
            db_user = db_config.get("USER", "debug")
            db_password = db_config.get("PASSWORD", "debug")

            connection_string = (
                f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
            logger.info(
                f"Database connection configured for: postgresql://<user>:<password>@{db_host}:{db_port}/{db_name}"
            )

            self.db_connector = PgConnector(connection_string)
            self.db_handler = PgHandler(self.db_connector)

            try:
                self.db_dialect = self.db_connector.dialect
            except AttributeError:
                self.db_dialect = DatabaseDialects.SQL
                logger.warning(
                    f"Connector type {type(self.db_connector)} does not have a 'dialect' attribute. "
                    f"Defaulting to '{self.db_dialect}'."
                )

            logger.info(
                f"Database components initialized with dialect: {self.db_dialect}"
            )
        except KeyError:
            logger.exception("Database 'default' configuration not found in settings.")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error initializing database components: {e}")
            raise

    def _initialize_model_registry(self):
        """Initializes the model registry and populates `allowed_tables`.

        Retrieves the global model registry instance. If it hasn't been initialized,
        it initializes it using the `ALLOWED_MODELS` setting from Django settings.
        Then, it fetches the list of all registered table names and stores them
        in `self.allowed_tables` for use in schema fetching and SQL safety checks.

        Raises:
            Exception: If registry initialization or table retrieval fails.
        """
        try:
            registry = get_registry()
            if not registry.initialized:
                model_specs = getattr(settings, "ALLOWED_MODELS", [])
                registry.initialize(model_specs)

            self.allowed_tables = registry.get_all_tables()
            if not self.allowed_tables:
                logger.warning(
                    "No allowed tables found in the model registry. SQL generation might be limited."
                )
            else:
                logger.info(
                    f"Allowed tables from model registry: {', '.join(self.allowed_tables)}"
                )
            logger.info("Model registry initialized.")
        except Exception as e:
            logger.exception(f"Error initializing model registry: {e}")
            raise

    def _initialize_llm_adapter(self):
        """Initializes the adapter for communication with the Large Language Model.

        Calls the factory method `LLMAdapter.get_adapter()` to obtain the configured
        LLM adapter instance (e.g., OpenAI, Gemini).

        Raises:
            Exception: If adapter instantiation fails.
        """
        try:
            self.llm_adapter = LLMAdapter.get_adapter()
            logger.info("LLM Adapter initialized.")
        except Exception as e:
            logger.exception(f"Error initializing LLM Adapter: {e}")
            raise

    def _initialize_conversation_storage(self):
        """Initializes the backend system for storing and retrieving conversation history.

        Calls the factory method `get_conversation_storage()` to obtain the configured
        storage instance (e.g., Redis, MongoDB). Also retrieves the
        `CONVERSATION_CONTEXT_LIMIT` from Django settings.

        Raises:
            Exception: If storage initialization fails.
        """
        try:
            self.conversation_storage = get_conversation_storage()
            self.context_limit = getattr(settings, "CONVERSATION_CONTEXT_LIMIT", 10)
            logger.info(
                f"Conversation storage initialized with context limit: {self.context_limit}"
            )
        except Exception as e:
            logger.exception(f"Error initializing conversation storage: {e}")
            raise

    def get_db_schema(self) -> str:
        """
        Fetches the database schema description for the tables allowed by the registry.

        Delegates the call to the `db_handler`, passing the list of allowed table names.
        This schema information is used to provide context to the LLM for SQL generation.

        Returns:
            str: A string describing the schema of the allowed tables (e.g., formatted
                 as CREATE TABLE statements or a custom description), or an error
                 message string starting with "Error:".
        """
        logger.info(f"Fetching schema for tables: {', '.join(self.allowed_tables)}")
        try:
            schema_info = self.db_handler.get_schema(self.allowed_tables)
            if isinstance(schema_info, str) and schema_info.startswith("Error:"):
                logger.error(f"Failed to get schema: {schema_info}")
            return schema_info
        except Exception as e:
            logger.exception(f"Unexpected error fetching DB schema: {e}")
            return f"Error: An unexpected error occurred while fetching the database schema: {e}"

    def execute_sql_query(self, sql_query: str) -> str:
        """
        Executes a given read-only SQL query against the database via the handler.

        Logs a truncated version of the query being executed.
        Delegates the execution to `self.db_handler.execute_query`.

        Args:
            sql_query (str): The SQL query string to execute.

        Returns:
            str: The formatted query results as a string (typically a markdown table),
                 or an error message string starting with "Error:" if execution failed.
        """
        logger.info(f"Attempting to execute SQL query: {sql_query[:100]}...")
        try:
            result = self.db_handler.execute_query(sql_query)
            if isinstance(result, str) and result.startswith("Error:"):
                logger.error(f"SQL execution failed: {result}")
            else:
                logger.info("SQL query executed successfully.")
            return result
        except Exception as e:
            logger.exception(f"Unexpected error executing SQL query: {e}")
            return f"Error: An unexpected error occurred during SQL execution: {e}"

    def create_conversation(self) -> str:
        """
        Creates a new, empty conversation entry in the configured storage backend.

        Delegates the call to `self.conversation_storage.create_conversation()`.

        Returns:
            str: The unique identifier (ID) for the newly created conversation.

        Raises:
            Exception: If the storage backend fails to create the conversation.
        """
        try:
            conv_id = self.conversation_storage.create_conversation()
            logger.info(f"Created new conversation with ID: {conv_id}")
            return conv_id
        except Exception as e:
            logger.exception(f"Error creating conversation: {e}")
            raise

    def save_message(self, conversation_id: str, role: str, content: str) -> bool:
        """
        Saves a single message entry to the history of a specified conversation.

        Constructs a message dictionary including role, content, and an ISO format timestamp.
        Delegates the saving operation to `self.conversation_storage.save_message()`.

        Args:
            conversation_id (str): The unique ID of the conversation to add the message to.
            role (str): The role of the message author (e.g., 'user', 'assistant', 'system').
            content (str): The textual content of the message.

        Returns:
            bool: True if the message was saved successfully by the storage backend,
                  False otherwise.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            success = self.conversation_storage.save_message(conversation_id, message)
            if not success:
                logger.error(
                    f"Failed to save message to conversation {conversation_id}"
                )
            # If RAG is enabled, also upsert vector asynchronously
            if (
                success
                and self.rag_enabled
                and self.vector_storage
                and self.embedding_model
            ):
                # Only embed non-system messages with content
                if role != "system" and content:
                    try:
                        vector = self.embedding_model.embed(content)
                        # Use sync_to_async if vector_storage is not async-native
                        upsert_async = sync_to_async(
                            self.vector_storage.upsert_message, thread_sensitive=False
                        )
                        asyncio.create_task(
                            upsert_async(conversation_id, message, vector)
                        )
                    except Exception as vector_exc:
                        logger.exception(
                            "Failed to embed or upsert message vector: %s", vector_exc
                        )
            return success
        except Exception as e:
            logger.exception(
                f"Error saving message to conversation {conversation_id}: {e}"
            )
            return False

    def _build_messages_for_llm(
        self, conversation_id: str, current_user_query: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Constructs messages for LLM, using RAG if enabled, falling back to window/summary.
        """
        try:
            if (
                self.rag_enabled
                and self.vector_storage
                and self.embedding_model
                and current_user_query
            ):
                # RAG Path
                logger.debug(
                    "Building context using RAG (k=%d, m=%d)",
                    self.rag_k,
                    self.rag_m_recent,
                )

                # 1. Get most recent M messages (for short-term memory/flow)
                recent_history = self.conversation_storage.get_conversation(
                    conversation_id, limit=self.rag_m_recent
                )

                # 2. Get top K relevant messages from vector store
                query_vector = self.embedding_model.embed(current_user_query)
                relevant_history = self.vector_storage.search_relevant(
                    conversation_id, query_vector, k=self.rag_k
                )

                # 3. Combine and deduplicate
                combined_history = []
                seen_timestamps = set()
                # Add relevant first, then recent, ensuring uniqueness by timestamp
                for msg in relevant_history + recent_history:
                    ts = msg.get("timestamp")
                    if ts and ts not in seen_timestamps:
                        combined_history.append(msg)
                        seen_timestamps.add(ts)

                # Sort combined by timestamp
                combined_history.sort(key=lambda x: x.get("timestamp", ""))

                # Filter out system messages and format for LLM
                messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in combined_history
                    if msg.get("role") != "system"
                ]

            else:
                # Fallback: Window + Summarization Path (Phase 2 logic)
                logger.debug("Building context using Sliding Window / Summarization")
                history = self.conversation_storage.get_conversation(conversation_id)

                if (
                    self.summarization_enabled
                    and len(history) > self.summarization_trigger
                ):
                    # (Existing Summarization logic from Phase 2 - keep as is)
                    summary_already_present = any(
                        m.get("role") == "system" and m.get("meta") == "summary"
                        for m in history
                    )
                    if not summary_already_present:
                        older_msgs = history[: -self.context_limit]
                        older_text = "\n".join(
                            f"{m['role']}: {m['content']}"
                            for m in older_msgs
                            if m.get("content")
                        )[:4000]
                        try:
                            summary_prompt = (
                                "Summarize the following conversation between a user and a database assistant in 2‑3 sentences. "
                                "Focus on the key facts, user intents, and any conclusions drawn.\n\n"
                                + older_text
                            )
                            summary_text = self.llm_adapter.generate_text(
                                "You are a helpful assistant that creates concise summaries.",
                                [{"role": "user", "content": summary_prompt}],
                            ).strip()
                            self.conversation_storage.save_message(
                                conversation_id,
                                {
                                    "role": "system",
                                    "content": summary_text,
                                    "meta": "summary",
                                },
                            )
                            history = self.conversation_storage.get_conversation(
                                conversation_id, limit=self.context_limit + 1
                            )
                        except Exception:
                            logger.exception(
                                "Failed to generate or save summary; falling back to trim."
                            )
                            history = history[-self.context_limit :]
                    else:
                        # If summary already present, just trim to context window + summary
                        summary_item = [
                            m for m in history if m.get("meta") == "summary"
                        ]
                        recent_items = history[-self.context_limit :]
                        history = summary_item + recent_items
                else:
                    # If no summarization needed yet, just limit by context_limit
                    history = history[-self.context_limit :]

                # Format for LLM, filtering system messages
                messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in history
                    if msg.get("role") != "system"
                ]

            # Append the current user query *after* history processing
            if current_user_query:
                messages.append({"role": "user", "content": current_user_query})

            return messages

        except Exception as e:
            logger.exception(
                f"Error building messages for LLM for {conversation_id}: {e}"
            )
            # Fallback to just the current query if history fails
            messages = []
            if current_user_query:
                messages.append({"role": "user", "content": current_user_query})
            return messages

    def _get_or_create_conversation(
        self, conversation_id: Optional[str]
    ) -> Optional[str]:
        if conversation_id is None:
            try:
                return self.create_conversation()
            except Exception:
                return None
        try:
            if not self.conversation_storage.conversation_exists(conversation_id):
                conversation_id = self.create_conversation()
            else:
                self.conversation_storage.update_expiry(conversation_id)
        except Exception:
            return None
        return conversation_id

    def _get_schema_or_error(self, conversation_id: str) -> Optional[str]:
        schema = self.get_db_schema()
        if isinstance(schema, str) and schema.startswith("Error:"):
            error_message = f"Sorry, I encountered an issue accessing the database structure needed to answer your question. {schema}"
            self.save_message(conversation_id, "assistant", error_message)
            return error_message
        return schema

    def _conversation_and_schema(self, user_query, conversation_id):
        """
        Ensure a conversation exists, save the user query, and fetch schema or return a schema error.
        Returns a tuple (conversation_id, schema, error_dict).
        """
        conversation_id = self._get_or_create_conversation(conversation_id)
        if conversation_id is None:
            return (
                None,
                None,
                {
                    "reply": "Sorry, I couldn't start a new conversation due to a system error.",
                    "conversation_id": None,
                    "error": "conversation_creation_error",
                },
            )
        self.save_message(conversation_id, "user", user_query)
        schema = self._get_schema_or_error(conversation_id)
        if isinstance(schema, str) and schema.startswith(
            "Sorry, I encountered an issue"
        ):
            return (
                conversation_id,
                None,
                {
                    "reply": schema,
                    "conversation_id": conversation_id,
                    "error": "schema_error",
                },
            )
        return conversation_id, schema, None

    def handle_query(
        self, user_query: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        conversation_id, schema, error = self._conversation_and_schema(
            user_query, conversation_id
        )
        if error:
            return error
        sql_result = self._generate_and_execute_sql(user_query, schema, conversation_id)
        if "error" in sql_result and "reply" in sql_result:
            return sql_result
        if "sql_query" in sql_result and "raw_result" in sql_result:
            raw_db_result = sql_result["raw_result"]
            executed_sql = sql_result["sql_query"]
            if isinstance(raw_db_result, str) and raw_db_result.startswith("Error:"):
                return self.handle_sql_error(
                    user_query, executed_sql, raw_db_result, conversation_id
                )
            else:
                return self._interpret_sql_results(
                    user_query, executed_sql, raw_db_result, schema, conversation_id
                )
        fallback_error = (
            "Sorry, an unexpected internal error occurred while processing your query."
        )
        self.save_message(conversation_id, "assistant", fallback_error)
        return {
            "reply": fallback_error,
            "conversation_id": conversation_id,
            "error": "internal_processing_error",
        }

    def _generate_and_execute_sql(
        self, user_query: str, schema: str, conversation_id: str
    ) -> Dict[str, Any]:
        """
        Attempts to generate SQL from a user query, delegates validation, and executes it.

        This involves:
        1. Building the LLM prompt with context.
        2. Calling the LLM to generate SQL.
        3. Cleaning the generated text.
        4. Calling `_validate_generated_sql` for validation (refusal, structure, safety, syntax).
        5. If validation passes, executing the query via `execute_sql_query()`.
        6. Saving the executed SQL and raw result to history.

        Args:
            user_query (str): The user's natural language query.
            schema (str): A string description of the relevant database schema.
            conversation_id (str): The ID of the current conversation for context.

        Returns:
            Dict[str, Any]:
            - On successful execution: {'sql_query': str, 'raw_result': Any}.
            - On validation failure or LLM refusal: An error dictionary from `_validate_generated_sql`.
            - On unexpected exception during generation/execution: An error dictionary.
        """
        try:
            conversation_messages = self._build_messages_for_llm(conversation_id)

            system_prompt = prompts.get_sql_generation_system_prompt(
                schema, self.allowed_tables, self.db_dialect
            )
            user_prompt = prompts.get_sql_generation_user_prompt(user_query)

            messages_for_llm = conversation_messages + [
                {"role": "user", "content": user_prompt}
            ]

            # Estimate and log token count before sending
            try:
                # Try to get actual model name from adapter if available
                llm_model_name = getattr(self.llm_adapter, "model_name", "gpt-4")
                token_estimate = _estimate_token_count(messages_for_llm, llm_model_name)
                logger.info(
                    f"Estimated token count for generate_text: {token_estimate}"
                )
            except Exception:
                logger.warning("Could not estimate token count.")

            logger.info("Generating SQL query via LLM...")
            generated_text = self.llm_adapter.generate_text(
                system_prompt, messages_for_llm
            )
            logger.info(f"LLM response received: '{generated_text[:100]}...'")

            sql_query = clean_sql_query(generated_text)
            logger.info(f"Cleaned SQL query: {sql_query[:100]}...")

            validation_error = self._validate_generated_sql(sql_query, conversation_id)
            if validation_error:
                return validation_error

            if not sql_query.endswith(";"):
                sql_query += ";"

            logger.info(f"Executing validated SQL query: {sql_query}")
            raw_result = self.execute_sql_query(sql_query)

            self.save_message(
                conversation_id,
                "system",
                f"Executed SQL: {sql_query}\nRaw Result: {raw_result}",
            )

            return {"sql_query": sql_query, "raw_result": raw_result}

        except Exception as e:
            logger.exception(
                f"Unexpected error during SQL generation or execution: {e}"
            )
            error_message = f"Sorry, I encountered an unexpected problem while preparing to answer your request. Error: {str(e)}"
            self.save_message(conversation_id, "assistant", error_message)
            return {
                "reply": error_message,
                "conversation_id": conversation_id,
                "error": "generation_execution_exception",
            }

    def _validate_generated_sql(
        self, sql_query: str, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Validates the generated SQL query for safety, structure, and common errors.

        Checks for:
        - Explicit refusal ("cannot answer").
        - Basic structural validity (e.g., starts with SELECT).
        - Safety (uses allowed tables, avoids forbidden operations).
        - Syntax errors (e.g., trailing commas).

        If validation fails, the error message is saved to the conversation history.

        Args:
            sql_query (str): The cleaned SQL query generated by the LLM.
            conversation_id (str): The ID of the current conversation for saving error messages.

        Returns:
            Optional[Dict[str, Any]]: An error dictionary if validation fails, suitable for
                                      returning directly from the calling method. Returns None
                                      if the query is valid.
        """
        if not sql_query:
            return {
                "error": "invalid_structure",
                "message": "No SQL query was generated.",
            }
        if "cannot answer" in sql_query.lower():
            return {
                "error": "cannot_answer",
                "message": "The system could not generate a SQL query for this question.",
            }
        if not is_valid_sql_structure(sql_query):
            error_message = "I couldn't generate a query with the correct structure (e.g., must start with SELECT). Please try rephrasing."
            logger.warning(f"Generated SQL failed structure validation: {sql_query}")
            self.save_message(conversation_id, "assistant", error_message)
            return {
                "reply": error_message,
                "conversation_id": conversation_id,
                "error": "invalid_structure",
            }

        if not is_safe_sql(sql_query, self.allowed_tables):
            cte_debug_info = ""
            if "with" in sql_query.lower():
                cte_matches = re.findall(
                    r"(\w+)\s+as\s*\(", sql_query.lower(), re.IGNORECASE
                )
                if cte_matches:
                    cte_debug_info = f" Query contains CTEs: {', '.join(cte_matches)}."

            error_message = "The generated query attempts to access data or use operations that are not permitted. Please focus your question on the allowed tables."
            logger.warning(f"Generated SQL failed safety validation: {sql_query}")
            logger.warning(f"Allowed tables: {', '.join(self.allowed_tables)}")
            if cte_debug_info:
                logger.warning(f"CTEs found: {cte_debug_info}")

            self.save_message(conversation_id, "assistant", error_message)
            return {
                "reply": error_message,
                "conversation_id": conversation_id,
                "error": "security_violation",
                "sql_query": sql_query,
            }

        if self._has_trailing_comma(sql_query):
            error_message = "I seem to have generated a query with a syntax error (a trailing comma). Could you please rephrase your question?"
            logger.warning(f"Generated SQL has trailing comma: {sql_query}")
            self.save_message(conversation_id, "assistant", error_message)
            return {
                "reply": error_message,
                "sql_query": sql_query,
                "raw_result": "Error: SQL query syntax error (trailing comma)",
                "conversation_id": conversation_id,
                "error": "trailing_comma",
            }

        return None

    def _has_trailing_comma(self, sql_query: str) -> bool:
        """Checks for a comma immediately preceding a closing parenthesis or semicolon
        or at the very end of the cleaned query string.

        Args:
            sql_query (str): The cleaned SQL query string.

        Returns:
            bool: True if a problematic trailing comma is found, False otherwise.
        """
        # Matches ',' followed by optional whitespace, then either ')', ';', or end-of-string
        return bool(re.search(r",\s*(\)|;|$)", sql_query.strip()))

    def _interpret_sql_results(
        self,
        user_query: str,
        sql_query: str,
        raw_result: Any,
        schema: str,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """Uses the LLM to interpret raw SQL query results into a natural language response.

        Takes the successful raw result from the database (which could be a list of dicts,
        a simple value, or a success message) and asks the LLM to explain it in the
        context of the original user query and the SQL that was executed.

        Args:
            user_query (str): The original natural language query from the user.
            sql_query (str): The SQL query that was successfully executed.
            raw_result (Any): The raw data or message returned by the database handler
                after successful query execution.
            schema (str): A string description of the relevant database schema for context.
            conversation_id (str): The ID of the current conversation.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'reply' (str): The generated natural language explanation.
                - 'sql_query' (str): The SQL query that was executed.
                - 'raw_result' (Any): The raw result from the database.
                - 'conversation_id' (str): The conversation ID.
                - 'error' (Optional[str]): 'interpretation_exception' if LLM call fails.
        """
        try:
            interpretation_messages = self._build_messages_for_llm(conversation_id)

            system_prompt = prompts.get_interpretation_system_prompt(schema, user_query)
            user_prompt = prompts.get_interpretation_user_prompt(
                user_query,
                sql_query,
                str(raw_result),
            )

            messages_for_llm = interpretation_messages + [
                {"role": "user", "content": user_prompt}
            ]

            # Estimate and log token count before sending
            try:
                llm_model_name = getattr(self.llm_adapter, "model_name", "gpt-4")
                token_estimate = _estimate_token_count(messages_for_llm, llm_model_name)
                logger.info(
                    f"Estimated token count for interpretation: {token_estimate}"
                )
            except Exception:
                logger.warning("Could not estimate token count for interpretation.")

            logger.info("Interpreting SQL results via LLM...")
            nl_response = self.llm_adapter.generate_text(
                system_prompt, messages_for_llm
            )
            logger.info("LLM interpretation received.")

            self.save_message(conversation_id, "assistant", nl_response)

            return {
                "reply": nl_response,
                "sql_query": sql_query,
                "raw_result": raw_result,
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.exception(f"Error interpreting SQL results: {e}")
            error_message = (
                f"I was able to retrieve the data, but encountered an issue while trying to explain it."
                f"\n\nExecuted SQL:\n```sql\n{sql_query}\n```\nRaw result:\n```\n{raw_result}\n```"
            )
            self.save_message(conversation_id, "assistant", error_message)
            return {
                "reply": error_message,
                "sql_query": sql_query,
                "raw_result": raw_result,
                "conversation_id": conversation_id,
                "error": "interpretation_exception",
            }

    def handle_sql_error(
        self, user_query: str, sql_query: str, error_message: str, conversation_id: str
    ) -> Dict[str, Any]:
        """Attempts to explain an SQL execution error in natural language using the LLM.

        When `execute_sql_query` returns an error string from the database handler,
        this method is called to ask the LLM to translate the technical error
        into a more user-friendly explanation.

        Args:
            user_query (str): The original natural language query from the user.
            sql_query (str): The SQL query that failed during execution.
            error_message (str): The error message string returned by the database handler.
            conversation_id (str): The ID of the current conversation.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'reply' (str): The generated natural language explanation of the error.
                - 'sql_query' (str): The SQL query that failed.
                - 'raw_result' (str): The original error message from the database handler.
                - 'conversation_id' (str): The conversation ID.
                - 'error' (str): Always 'sql_execution_error' for this path, plus
                  'error_explanation_exception' if the LLM call itself fails.
        """
        try:
            logger.info(f"Handling SQL execution error: {error_message}")
            error_context_messages = self._build_messages_for_llm(conversation_id)

            system_prompt = prompts.get_error_system_prompt()
            user_prompt = prompts.get_error_user_prompt(sql_query, error_message)

            messages_for_llm = error_context_messages + [
                {"role": "user", "content": user_prompt}
            ]

            # Estimate and log token count before sending
            try:
                llm_model_name = getattr(self.llm_adapter, "model_name", "gpt-4")
                token_estimate = _estimate_token_count(messages_for_llm, llm_model_name)
                logger.info(
                    f"Estimated token count for error explanation: {token_estimate}"
                )
            except Exception:
                logger.warning("Could not estimate token count for error explanation.")

            logger.info("Generating friendly SQL error explanation via LLM...")
            friendly_error = self.llm_adapter.generate_text(
                system_prompt, messages_for_llm
            )
            logger.info("LLM error explanation received.")

            self.save_message(conversation_id, "assistant", friendly_error)

            return {
                "reply": friendly_error,
                "sql_query": sql_query,
                "raw_result": error_message,
                "conversation_id": conversation_id,
                "error": "sql_execution_error",
            }

        except Exception as e:
            logger.exception(f"Error generating SQL error explanation: {e}")
            fallback_error_message = f"I encountered an error trying to execute the query.\n\nDetails: {error_message}"
            self.save_message(conversation_id, "assistant", fallback_error_message)
            return {
                "reply": fallback_error_message,
                "sql_query": sql_query,
                "raw_result": error_message,
                "conversation_id": conversation_id,
                "error": "error_explanation_exception",
            }

    async def handle_query_stream(
        self,
        user_query: str,
        conversation_id: Optional[str] = None,
        status_callback: Optional[Any] = None,
    ):
        """
        Streams the response to a user query over a websocket, sending status updates at each backend step.
        Args:
            user_query (str): The user's natural language query.
            conversation_id (Optional[str]): The conversation ID for context.
            status_callback (Optional[Callable]): Async function to send status updates to the frontend.
        Yields:
            dict: Websocket message chunks (llm_token, llm_stream_end, or error).
        """
        # Conversation and schema initialization
        if status_callback:
            await status_callback("Generating query SQL...")
        conversation_id, schema, error = self._conversation_and_schema(
            user_query, conversation_id
        )
        if error:
            if status_callback:
                await status_callback("Preparing error response…")
            err_type = error.get("error", "generic_error")
            handler = get_handler(err_type)
            async for chunk in handler.stream(
                self,
                user_query,
                "",
                error["reply"],
                schema if schema else "",
                conversation_id,
                status_callback=status_callback,
            ):
                yield chunk
            return
        # Generate and execute SQL
        if status_callback:
            await status_callback("Validating and executing SQL...")
        sql_result = self._generate_and_execute_sql(user_query, schema, conversation_id)
        # Centralised error handling --------------------------------------
        if "error" in sql_result:
            error_type = sql_result.get("error", "generic_error")
            executed_sql = sql_result.get("sql_query", "")
            error_message = sql_result.get("reply", "Unknown error")
            handler = get_handler(error_type)
            async for chunk in handler.stream(
                self,
                user_query,
                executed_sql,
                error_message,
                schema,
                conversation_id,
                status_callback=status_callback,
            ):
                yield chunk
            return
        # Successful SQL; handle execution errors vs interpretation
        if "sql_query" in sql_result and "raw_result" in sql_result:
            raw_db_result = sql_result["raw_result"]
            executed_sql = sql_result["sql_query"]
            if isinstance(raw_db_result, str) and raw_db_result.startswith("Error:"):
                if status_callback:
                    await status_callback(
                        "SQL execution error. Converting error to human readable message..."
                    )
                handler = get_handler("sql_execution_error")
                async for chunk in handler.stream(
                    self,
                    user_query,
                    executed_sql,
                    raw_db_result,
                    schema,
                    conversation_id,
                    status_callback=status_callback,
                ):
                    yield chunk
                return
            if status_callback:
                await status_callback(
                    "Converting SQL result to human readable message..."
                )
            interpretation_messages = self._build_messages_for_llm(conversation_id)
            system_prompt = prompts.get_interpretation_system_prompt(schema, user_query)
            user_prompt = prompts.get_interpretation_user_prompt(
                user_query, executed_sql, str(raw_db_result)
            )
            messages_for_llm = interpretation_messages + [
                {"role": "user", "content": user_prompt}
            ]

            # Estimate and log token count before streaming
            try:
                llm_model_name = getattr(self.llm_adapter, "model_name", "gpt-4")
                token_estimate = _estimate_token_count(messages_for_llm, llm_model_name)
                logger.info(
                    f"Estimated token count for stream_text (interpretation): {token_estimate}"
                )
            except Exception:
                logger.warning("Could not estimate token count for stream_text.")

            full_message = ""
            try:
                async for chunk in self.llm_adapter.stream_text(
                    system_prompt, messages_for_llm
                ):
                    full_message += chunk
                    yield {
                        "type": "llm_token",
                        "token": chunk,
                        "conversation_id": conversation_id,
                    }
                yield {
                    "type": "llm_stream_end",
                    "message": full_message,
                    "conversation_id": conversation_id,
                }
                self.save_message(conversation_id, "assistant", full_message)
            except Exception as e:
                if status_callback:
                    await status_callback(
                        "Error during streaming of LLM interpretation."
                    )
                yield {
                    "type": "error",
                    "message": f"Streaming error: {str(e)}",
                    "conversation_id": conversation_id,
                }
                return
        else:
            if status_callback:
                await status_callback(
                    "Unexpected internal error. Generating error message..."
                )
            fallback_error = "Sorry, an unexpected internal error occurred while processing your query."
            self.save_message(conversation_id, "assistant", fallback_error)
            handler = get_handler("internal_processing_error")
            async for chunk in handler.stream(
                self,
                user_query,
                "",
                fallback_error,
                schema if schema else "",
                conversation_id,
                status_callback=status_callback,
            ):
                yield chunk
            return
