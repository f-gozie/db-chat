"""Core service for the database chat application."""

import json
import logging
import re
from datetime import datetime

from django.conf import settings

from . import prompts
from .db_executor import PostgreSQLExecutor
from .llm_adapter import LLMAdapter
from .model_registry import get_registry
from .storage import get_conversation_storage
from .utils import clean_sql_query, is_safe_sql, is_valid_sql_structure

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat interactions with the database via LLM."""

    def __init__(self):
        self._initialize_db_executor()
        self._initialize_model_registry()
        self._initialize_llm_adapter()
        self._initialize_conversation_storage()

    def _initialize_db_executor(self):
        """Initialize the database executor."""
        db_host = settings.DATABASES["default"].get("HOST", "postgres")
        db_port = settings.DATABASES["default"].get("PORT", "5432")
        db_name = settings.DATABASES["default"].get("NAME", "eyemark_backend")
        db_user = settings.DATABASES["default"].get("USER", "debug")
        db_password = settings.DATABASES["default"].get("PASSWORD", "debug")

        connection_string = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
        logger.info(
            f"Using database connection string pattern: postgresql://user:password@{db_host}:{db_port}/{db_name}"
        )

        self.db_executor = PostgreSQLExecutor(connection_string)

    def _initialize_model_registry(self):
        """Initialize the model registry and get allowed tables."""
        registry = get_registry()
        if not registry.initialized:
            model_specs = getattr(settings, "ALLOWED_MODELS", [])
            registry.initialize(model_specs)

        self.allowed_tables = registry.get_all_tables()
        logger.info(
            f"Allowed tables from model registry: {', '.join(self.allowed_tables)}"
        )

    def _initialize_llm_adapter(self):
        """Initialize the LLM adapter."""
        self.llm_adapter = LLMAdapter.get_adapter()

    def _initialize_conversation_storage(self):
        """Initialize conversation storage."""
        self.conversation_storage = get_conversation_storage()
        self.context_limit = getattr(settings, "CONVERSATION_CONTEXT_LIMIT", 10)

    def get_db_schema(self):
        """Fetch schema of allowed tables using direct PostgreSQL connection."""
        logger.info(f"Fetching schema for tables: {', '.join(self.allowed_tables)}")
        return self.db_executor.get_schema(self.allowed_tables)

    def execute_sql_query(self, sql_query: str):
        """Execute a read-only SQL query using direct PostgreSQL connection."""
        logger.info(f"Executing SQL query: {sql_query}")
        return self.db_executor.execute_query(sql_query)["result"]

    def create_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        return self.conversation_storage.create_conversation()

    def save_message(self, conversation_id: str, role: str, content: str) -> bool:
        """Save a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": json.dumps(
                {"$date": {"$numberLong": str(int(datetime.now().timestamp() * 1000))}}
            ),
        }
        return self.conversation_storage.save_message(conversation_id, message)

    def _build_messages_for_llm(
        self, conversation_id: str, user_query: str = ""
    ) -> list:
        """Build message array for the LLM including conversation history."""
        history = self.conversation_storage.get_conversation(
            conversation_id, limit=self.context_limit
        )

        messages = []
        for msg in history:
            if msg["role"] != "system":
                messages.append({"role": msg["role"], "content": msg["content"]})

        if user_query:
            messages.append({"role": "user", "content": user_query})

        return messages

    def handle_query(self, user_query: str, conversation_id: str = None):
        """Handle a user query by generating and executing SQL, then interpreting results."""
        logger.info(f"Handling user query: {user_query}")

        # Create or validate conversation ID
        if conversation_id is None:
            conversation_id = self.conversation_storage.create_conversation()
            logger.info(f"Created new conversation: {conversation_id}")
        else:
            self.conversation_storage.update_expiry(conversation_id)
            logger.info(f"Using existing conversation: {conversation_id}")

        # Save user query to conversation history
        self.save_message(conversation_id, "user", user_query)

        # Get DB schema for context
        schema = self.get_db_schema()
        if isinstance(schema, str) and schema.startswith("Error:"):
            error_message = f"Error fetching database schema: {schema}"
            self.save_message(conversation_id, "assistant", error_message)
            return {"reply": error_message, "conversation_id": conversation_id}

        # Generate and execute SQL query
        sql_result = self._generate_and_execute_sql(user_query, schema, conversation_id)
        if "error" in sql_result:
            return sql_result

        # Interpret results and return response
        return self._interpret_sql_results(
            user_query,
            sql_result["sql_query"],
            sql_result["raw_result"],
            schema,
            conversation_id,
        )

    def _generate_and_execute_sql(self, user_query, schema, conversation_id):
        """Generate SQL from user query and execute it."""
        try:
            # Get conversation context
            conversation_messages = self._build_messages_for_llm(conversation_id)

            # Generate SQL query
            system_prompt = prompts.get_sql_generation_system_prompt(
                schema, self.allowed_tables
            )
            user_prompt = prompts.get_sql_generation_user_prompt(user_query)

            messages = conversation_messages + [
                {"role": "user", "content": user_prompt}
            ]
            generated_text = self.llm_adapter.generate_text(system_prompt, messages)

            # Clean and validate SQL query
            combined_sql = clean_sql_query(generated_text)

            # Check for "cannot answer" response
            if combined_sql.startswith(
                "I cannot answer this query with the available data and tools."
            ):
                self.save_message(conversation_id, "assistant", combined_sql)
                return {
                    "reply": combined_sql,
                    "conversation_id": conversation_id,
                    "error": "cannot_answer",
                }

            # Validate SQL structure
            if not is_valid_sql_structure(combined_sql):
                error_message = "I couldn't generate a valid SQL query for your question. Please try rephrasing or ask about data that's available in the database schema."
                self.save_message(conversation_id, "assistant", error_message)
                return {
                    "reply": error_message,
                    "conversation_id": conversation_id,
                    "error": "invalid_structure",
                }

            # Validate against security constraints
            if not is_safe_sql(combined_sql, self.allowed_tables):
                error_message = "I couldn't generate a safe SQL query for your question. Your query might reference tables that aren't available or contain operations that aren't permitted."
                self.save_message(conversation_id, "assistant", error_message)
                return {
                    "reply": error_message,
                    "conversation_id": conversation_id,
                    "error": "security_violation",
                }

            # Check for trailing commas
            if self._has_trailing_comma(combined_sql):
                error_message = "I tried to create a SQL query to answer your question, but it appears the query has a syntax error with a trailing comma. Please try rephrasing your question."
                self.save_message(conversation_id, "assistant", error_message)
                return {
                    "reply": error_message,
                    "sql_query": combined_sql,
                    "raw_result": "Error: SQL query has a trailing comma",
                    "conversation_id": conversation_id,
                    "error": "trailing_comma",
                }

            # Ensure query ends with a semicolon
            if not combined_sql.endswith(";"):
                combined_sql += ";"

            # Execute the validated SQL query
            sql_to_execute = combined_sql
            logger.info(f"Generated SQL query: {sql_to_execute}")

            raw_result = self.execute_sql_query(sql_to_execute)
            logger.info("SQL query executed successfully")

            return {"sql_query": sql_to_execute, "raw_result": raw_result}

        except Exception as e:
            logger.error(f"Error generating/executing SQL: {e}")
            error_message = f"Error: {str(e)}"
            self.save_message(conversation_id, "assistant", error_message)
            return {
                "reply": error_message,
                "conversation_id": conversation_id,
                "error": "execution_error",
            }

    def _has_trailing_comma(self, sql_query):
        """Check if SQL query has a trailing comma."""
        return bool(re.search(r",\s*$", sql_query) or re.search(r",\s*;$", sql_query))

    def _interpret_sql_results(
        self, user_query, sql_query, raw_result, schema, conversation_id
    ):
        """Interpret SQL results using LLM."""
        try:
            # Get updated conversation context
            interpretation_messages = self._build_messages_for_llm(conversation_id)

            # Generate natural language interpretation
            system_prompt = prompts.get_interpretation_system_prompt(schema, user_query)
            user_prompt = prompts.get_interpretation_user_prompt(
                user_query, sql_query, raw_result
            )

            messages = interpretation_messages + [
                {"role": "user", "content": user_prompt}
            ]
            nl_response = self.llm_adapter.generate_text(system_prompt, messages)

            # Store SQL query and result as system message for context
            self.save_message(
                conversation_id, "system", f"SQL: {sql_query}\nResult: {raw_result}"
            )

            # Save assistant response
            self.save_message(conversation_id, "assistant", nl_response)

            return {
                "reply": nl_response,
                "sql_query": sql_query,
                "raw_result": raw_result,
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.error(f"Error interpreting SQL results: {e}")
            # Fallback to raw results
            error_message = (
                f"Error interpreting results: {str(e)}\n\nRaw result: {raw_result}"
            )
            self.save_message(conversation_id, "assistant", error_message)
            return {
                "reply": error_message,
                "sql_query": sql_query,
                "raw_result": raw_result,
                "conversation_id": conversation_id,
            }

    def handle_sql_error(self, user_query, sql_query, error_message, conversation_id):
        """Handle SQL execution errors by explaining them in natural language."""
        try:
            # Get conversation context for error handling
            error_messages = self._build_messages_for_llm(conversation_id)

            # Generate error explanation
            system_prompt = prompts.get_error_system_prompt()
            user_prompt = prompts.get_error_user_prompt(sql_query, error_message)

            messages = error_messages + [{"role": "user", "content": user_prompt}]
            friendly_error = self.llm_adapter.generate_text(system_prompt, messages)

            # Save to conversation history
            self.save_message(conversation_id, "assistant", friendly_error)

            return {
                "reply": friendly_error,
                "sql_query": sql_query,
                "raw_result": f"Error executing SQL query: {error_message}",
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.error(f"Error generating SQL error explanation: {e}")
            # Fallback to simple error message
            error_message = f"Error executing SQL query: {error_message}"
            self.save_message(conversation_id, "assistant", error_message)
            return {
                "reply": error_message,
                "sql_query": sql_query,
                "conversation_id": conversation_id,
            }
