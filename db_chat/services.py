"""Core service for the database chat application."""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from django.conf import settings

from . import prompts
from .connectors.postgres.pg_connector import PgConnector
from .connectors.postgres.pg_handler import PgHandler
from .llm_adapter import LLMAdapter
from .model_registry import get_registry
from .storage import ConversationStorage, get_conversation_storage
from .utils import clean_sql_query, is_safe_sql, is_valid_sql_structure

logger = logging.getLogger(__name__)


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
    """

    db_connector: PgConnector
    db_handler: PgHandler
    llm_adapter: LLMAdapter
    conversation_storage: ConversationStorage
    allowed_tables: List[str]
    context_limit: int

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
            db_name = db_config.get("NAME", "eyemark_backend")
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
            logger.info("Database components initialized.")
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
        Constructs the list of messages to be sent to the LLM for context.

        Retrieves historical messages for the given conversation ID from the storage
        backend, limited by `self.context_limit`. It filters out any messages with
        the role 'system' (which might contain raw SQL/results not meant for direct
        LLM context in the next turn) and formats the remaining messages (user/assistant)
        into the standard dictionary format expected by the LLM adapter.
        Optionally appends the `current_user_query` as the latest user message.

        Args:
            conversation_id (str): The ID of the conversation whose history is needed.
            current_user_query (Optional[str]): The most recent query from the user, to be
                appended to the history as the last message. Defaults to None.

        Returns:
            List[Dict[str, str]]: A list of message dictionaries, ordered chronologically,
                                  ready to be passed to the LLM adapter.
                                  Returns an empty list if history retrieval fails.
        """
        try:
            history = self.conversation_storage.get_conversation(
                conversation_id, limit=self.context_limit
            )
        except Exception as e:
            logger.exception(
                f"Error retrieving conversation history for {conversation_id}: {e}"
            )
            history = []

        messages = []
        for msg in history:
            # Exclude system messages (often used for internal SQL/result logging)
            if msg.get("role") != "system":
                content = msg.get("content", "")
                if not isinstance(content, str):
                    logger.warning(
                        f"Non-string content found in message history for {conversation_id}, converting."
                    )
                    content = str(content)
                messages.append({"role": msg["role"], "content": content})

        if current_user_query:
            messages.append({"role": "user", "content": current_user_query})

        return messages

    def handle_query(
        self, user_query: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing a user's natural language query.

        This method orchestrates the entire workflow:
        1. Manages conversation context (creates a new ID or uses/validates existing one).
        2. Saves the incoming user query to the conversation history.
        3. Retrieves the database schema for the allowed tables.
        4. Calls `_generate_and_execute_sql` to get the LLM to produce and validate SQL,
           and then execute it.
        5. Based on the outcome of step 4:
           - If successful execution: Calls `_interpret_sql_results` to get the LLM
             to explain the results in natural language.
           - If SQL execution failed: Calls `handle_sql_error` to get the LLM to
             explain the database error.
           - If SQL generation/validation failed: Returns the error message directly.
        6. Returns a dictionary containing the final user-facing reply and other relevant
           details (conversation ID, SQL query, raw result, error code).

        Args:
            user_query (str): The natural language query submitted by the user.
            conversation_id (Optional[str]): The ID of the conversation to continue.
                If None, a new conversation is created. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary summarizing the outcome, including:
                - 'reply' (str): The final natural language response for the user.
                - 'conversation_id' (str): The ID of the conversation (new or existing).
                - 'sql_query' (Optional[str]): The SQL query generated and executed, if any.
                - 'raw_result' (Optional[Any]): The raw data returned by the query, if executed.
                - 'error' (Optional[str]): A short code indicating the type of error encountered,
                    if any (e.g., 'schema_error', 'invalid_structure', 'security_violation',
                    'sql_execution_error', 'interpretation_exception').
        """
        logger.info(
            f"Handling user query: '{user_query[:50]}...' in conversation: {conversation_id}"
        )

        # --- 1. Conversation Management ---
        if conversation_id is None:
            try:
                conversation_id = self.create_conversation()
            except Exception:
                return {
                    "reply": "Sorry, I couldn't start a new conversation due to a system error.",
                    "conversation_id": None,
                    "error": "conversation_creation_error",
                }
        else:
            try:
                if not self.conversation_storage.conversation_exists(conversation_id):
                    logger.warning(
                        f"Conversation ID {conversation_id} not found. Starting a new one."
                    )
                    conversation_id = self.create_conversation()
                else:
                    self.conversation_storage.update_expiry(conversation_id)
                    logger.info(f"Using existing conversation: {conversation_id}")
            except Exception as e:
                logger.exception(f"Error managing conversation {conversation_id}: {e}")
                # Attempt to continue, but log the issue.

        # --- 2. Save User Query ---
        self.save_message(conversation_id, "user", user_query)

        # --- 3. Get Schema ---
        schema = self.get_db_schema()
        if isinstance(schema, str) and schema.startswith("Error:"):
            error_message = f"Sorry, I encountered an issue accessing the database structure needed to answer your question. {schema}"
            self.save_message(conversation_id, "assistant", error_message)
            return {
                "reply": error_message,
                "conversation_id": conversation_id,
                "error": "schema_error",
            }

        # --- 4. Generate, Validate, and Execute SQL ---
        sql_result = self._generate_and_execute_sql(user_query, schema, conversation_id)

        # --- 5. Process SQL Result (Interpret, Explain Error, or Handle Generation Failure) ---
        if "error" in sql_result and "reply" in sql_result:
            # SQL generation/validation failed, error message already saved by _generate_and_execute_sql
            logger.warning(
                f"SQL generation/validation failed with error: {sql_result.get('error')}"
            )
            return sql_result

        if "sql_query" in sql_result and "raw_result" in sql_result:
            raw_db_result = sql_result["raw_result"]
            executed_sql = sql_result["sql_query"]

            if isinstance(raw_db_result, str) and raw_db_result.startswith("Error:"):
                # SQL execution failed
                logger.info(f"SQL execution resulted in an error: {raw_db_result}")
                return self.handle_sql_error(
                    user_query, executed_sql, raw_db_result, conversation_id
                )
            else:
                # SQL execution succeeded
                logger.info("SQL executed successfully, proceeding to interpretation.")
                return self._interpret_sql_results(
                    user_query, executed_sql, raw_db_result, schema, conversation_id
                )
        else:
            # This case should ideally not be reached if _generate_and_execute_sql returns correctly
            logger.error(
                "Internal inconsistency: _generate_and_execute_sql finished without error but missing expected keys."
            )
            fallback_error = "Sorry, an unexpected internal error occurred while processing your query."
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
        Attempts to generate SQL from a user query, validate it, and execute it.

        This involves:
        1. Building the LLM prompt with schema, allowed tables, history, and the user query.
        2. Calling the LLM to generate the SQL query text.
        3. Cleaning the generated text (removing markdown, extra characters).
        4. Validating the cleaned SQL for:
           - Explicit refusal ("cannot answer").
           - Basic structural validity (e.g., starts with SELECT).
           - Safety (uses allowed tables, avoids forbidden operations).
           - Syntax errors (e.g., trailing commas).
        5. If validation passes, appending a semicolon if needed.
        6. Executing the validated query using `self.execute_sql_query()`.
        7. Saving the *executed* SQL and its *raw* result (even if it's an error message
           from the DB) to conversation history with role 'system' for internal context.

        Args:
            user_query (str): The user's natural language query.
            schema (str): A string description of the relevant database schema.
            conversation_id (str): The ID of the current conversation for context.

        Returns:
            Dict[str, Any]:
            - On successful execution: Returns {'sql_query': str, 'raw_result': Any}
              (Note: 'raw_result' could still be an error string from the DB handler).
            - On validation failure or LLM refusal: Returns {'reply': str, 'conversation_id': str,
              'error': str, possibly 'sql_query' and 'raw_result' for logging context}.
            - On unexpected exception: Returns {'reply': str, 'conversation_id': str,
              'error': 'generation_execution_exception'}.
        """
        try:
            conversation_messages = self._build_messages_for_llm(conversation_id)

            system_prompt = prompts.get_sql_generation_system_prompt(
                schema, self.allowed_tables
            )
            user_prompt = prompts.get_sql_generation_user_prompt(user_query)

            messages_for_llm = conversation_messages + [
                {"role": "user", "content": user_prompt}
            ]

            logger.info("Generating SQL query via LLM...")
            generated_text = self.llm_adapter.generate_text(
                system_prompt, messages_for_llm
            )
            logger.info(f"LLM response received: '{generated_text[:100]}...'")

            sql_query = clean_sql_query(generated_text)
            logger.info(f"Cleaned SQL query: {sql_query[:100]}...")

            # --- Validation Steps ---
            if "cannot answer" in sql_query.lower():
                logger.warning(f"LLM indicated it cannot answer query: {sql_query}")
                self.save_message(conversation_id, "assistant", sql_query)
                return {
                    "reply": sql_query,
                    "conversation_id": conversation_id,
                    "error": "cannot_answer",
                }

            if not is_valid_sql_structure(sql_query):
                error_message = "I couldn't generate a query with the correct structure (e.g., must start with SELECT). Please try rephrasing."
                logger.warning(
                    f"Generated SQL failed structure validation: {sql_query}"
                )
                self.save_message(conversation_id, "assistant", error_message)
                return {
                    "reply": error_message,
                    "conversation_id": conversation_id,
                    "error": "invalid_structure",
                }

            if not is_safe_sql(sql_query, self.allowed_tables):
                error_message = "The generated query attempts to access data or use operations that are not permitted. Please focus your question on the allowed tables."
                logger.warning(f"Generated SQL failed safety validation: {sql_query}")
                self.save_message(conversation_id, "assistant", error_message)
                return {
                    "reply": error_message,
                    "conversation_id": conversation_id,
                    "error": "security_violation",
                }

            if self._has_trailing_comma(sql_query):
                error_message = "I seem to have generated a query with a syntax error (a trailing comma). Could you please rephrase your question?"
                logger.warning(f"Generated SQL has trailing comma: {sql_query}")
                self.save_message(conversation_id, "assistant", error_message)
                return {
                    "reply": error_message,
                    "sql_query": sql_query,  # Include faulty query for context
                    "raw_result": "Error: SQL query syntax error (trailing comma)",
                    "conversation_id": conversation_id,
                    "error": "trailing_comma",
                }

            # --- Execution Preparation ---
            if not sql_query.endswith(";"):
                sql_query += ";"

            # --- Execute Validated Query ---
            logger.info(f"Executing validated SQL query: {sql_query}")
            raw_result = self.execute_sql_query(sql_query)

            # --- Save Context for Future Turns (Internal) ---
            self.save_message(
                conversation_id,
                "system",
                f"Executed SQL: {sql_query}\nRaw Result: {raw_result}",
            )

            # Return execution details (raw_result might be data or an error string from DB)
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
            # Build context, including the latest user query implicitly via history
            interpretation_messages = self._build_messages_for_llm(conversation_id)

            system_prompt = prompts.get_interpretation_system_prompt(schema, user_query)
            # Provide the executed SQL and its result explicitly in the user prompt
            user_prompt = prompts.get_interpretation_user_prompt(
                user_query,
                sql_query,
                str(raw_result),  # Ensure result is stringified for prompt
            )

            messages_for_llm = interpretation_messages + [
                # Add SQL/Result context as a system message before the interpretation request
                # Note: This differs slightly from original; aiming for clarity for LLM.
                # If issues arise, revert to putting SQL/result only in user_prompt.
                {
                    "role": "system",
                    "content": f"Executed SQL:\n```sql\n{sql_query}\n```\nResult:\n```\n{raw_result}\n```",
                },
                {
                    "role": "user",
                    "content": "Based on the original question and the SQL execution details above, please provide a natural language explanation of the result.",
                },
            ]

            logger.info("Interpreting SQL results via LLM...")
            nl_response = self.llm_adapter.generate_text(
                system_prompt, messages_for_llm
            )
            logger.info("LLM interpretation received.")

            # Save the final natural language response from the assistant
            self.save_message(conversation_id, "assistant", nl_response)

            return {
                "reply": nl_response,
                "sql_query": sql_query,
                "raw_result": raw_result,
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.exception(f"Error interpreting SQL results: {e}")
            # Fallback: Provide the raw result if interpretation fails
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

            logger.info("Generating friendly SQL error explanation via LLM...")
            friendly_error = self.llm_adapter.generate_text(
                system_prompt, messages_for_llm
            )
            logger.info("LLM error explanation received.")

            # Save the friendly error explanation as the assistant's response
            self.save_message(conversation_id, "assistant", friendly_error)

            return {
                "reply": friendly_error,
                "sql_query": sql_query,
                "raw_result": error_message,  # The original DB error is the 'raw result' here
                "conversation_id": conversation_id,
                "error": "sql_execution_error",
            }

        except Exception as e:
            logger.exception(f"Error generating SQL error explanation: {e}")
            # Fallback: Just return the original technical error if explanation fails
            fallback_error_message = f"I encountered an error trying to execute the query.\n\nDetails: {error_message}"
            self.save_message(conversation_id, "assistant", fallback_error_message)
            return {
                "reply": fallback_error_message,
                "sql_query": sql_query,
                "raw_result": error_message,
                "conversation_id": conversation_id,
                "error": "error_explanation_exception",  # Indicate the explanation step failed
            }
