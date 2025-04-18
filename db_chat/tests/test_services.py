from unittest.mock import MagicMock, patch

import pytest

from db_chat.constants import DatabaseDialects
from db_chat.services import ChatService


@pytest.fixture
def mock_services():
    """Set up mocks for all service dependencies"""
    with patch("db_chat.services.PgConnector") as mock_connector, patch(
        "db_chat.services.PgHandler"
    ) as mock_handler, patch("db_chat.services.LLMAdapter") as mock_llm_adapter, patch(
        "db_chat.services.get_registry"
    ) as mock_get_registry, patch(
        "db_chat.services.get_conversation_storage"
    ) as mock_get_storage:

        mock_connector_instance = MagicMock()
        mock_connector_instance.dialect = DatabaseDialects.POSTGRESQL
        mock_connector.return_value = mock_connector_instance

        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance

        mock_llm_adapter_instance = MagicMock()
        mock_llm_adapter.get_adapter.return_value = mock_llm_adapter_instance

        mock_registry_instance = MagicMock()
        mock_registry_instance.initialized = True
        mock_registry_instance.get_all_tables.return_value = [
            "users",
            "orders",
            "products",
        ]
        mock_get_registry.return_value = mock_registry_instance

        mock_storage_instance = MagicMock()
        mock_storage_instance.create_conversation.return_value = "test-conversation-id"
        mock_get_storage.return_value = mock_storage_instance

        yield {
            "connector": mock_connector_instance,
            "handler": mock_handler_instance,
            "llm_adapter": mock_llm_adapter_instance,
            "registry": mock_registry_instance,
            "storage": mock_storage_instance,
        }


class TestChatService:
    """Tests for the ChatService class"""

    def test_initialization(self, mock_services):
        """Test that the service initializes correctly"""

        service = ChatService()

        assert service.db_connector == mock_services["connector"]
        assert service.db_handler == mock_services["handler"]
        assert service.llm_adapter == mock_services["llm_adapter"]
        assert service.allowed_tables == ["users", "orders", "products"]
        assert service.conversation_storage == mock_services["storage"]
        assert service.db_dialect == DatabaseDialects.POSTGRESQL

    def test_get_db_schema_success(self, mock_services):
        """Test getting DB schema successfully"""
        mock_services["handler"].get_schema.return_value = "Table: users (id, name)"

        service = ChatService()
        schema = service.get_db_schema()

        mock_services["handler"].get_schema.assert_called_once_with(
            ["users", "orders", "products"]
        )
        assert schema == "Table: users (id, name)"

    def test_get_db_schema_error(self, mock_services):
        """Test handling error when getting DB schema"""
        mock_services["handler"].get_schema.side_effect = Exception("Test error")

        service = ChatService()
        schema = service.get_db_schema()

        assert "Error:" in schema
        assert "Test error" in schema

    def test_execute_sql_query_success(self, mock_services):
        """Test successful SQL query execution"""
        mock_services["handler"].execute_query.return_value = "| id | name |\n|1|Test|"

        service = ChatService()
        result = service.execute_sql_query("SELECT * FROM users")

        mock_services["handler"].execute_query.assert_called_once_with(
            "SELECT * FROM users"
        )
        assert result == "| id | name |\n|1|Test|"

    def test_execute_sql_query_error(self, mock_services):
        """Test handling error in SQL query execution"""
        mock_services["handler"].execute_query.return_value = "Error: Invalid SQL"

        service = ChatService()
        result = service.execute_sql_query("INVALID SQL")

        assert result == "Error: Invalid SQL"

    def test_execute_sql_query_exception(self, mock_services):
        """Test handling exception in SQL query execution"""
        mock_services["handler"].execute_query.side_effect = Exception(
            "Unexpected error"
        )

        service = ChatService()
        result = service.execute_sql_query("SELECT * FROM users")

        assert "Error:" in result
        assert "Unexpected error" in result

    def test_create_conversation(self, mock_services):
        """Test creating a new conversation"""
        service = ChatService()
        conv_id = service.create_conversation()

        mock_services["storage"].create_conversation.assert_called_once()
        assert conv_id == "test-conversation-id"

    def test_save_message_success(self, mock_services):
        """Test saving a message successfully"""
        mock_services["storage"].save_message.return_value = True

        service = ChatService()
        result = service.save_message("test-conv-id", "user", "Test message")

        assert result is True
        mock_services["storage"].save_message.assert_called_once()
        assert mock_services["storage"].save_message.call_args[0][0] == "test-conv-id"
        assert mock_services["storage"].save_message.call_args[0][1]["role"] == "user"
        assert (
            mock_services["storage"].save_message.call_args[0][1]["content"]
            == "Test message"
        )

    def test_save_message_failure(self, mock_services):
        """Test handling failure when saving message"""
        mock_services["storage"].save_message.return_value = False

        service = ChatService()
        result = service.save_message("test-conv-id", "user", "Test message")

        assert result is False

    def test_save_message_exception(self, mock_services):
        """Test handling exception when saving message"""
        mock_services["storage"].save_message.side_effect = Exception("Storage error")

        service = ChatService()
        result = service.save_message("test-conv-id", "user", "Test message")

        assert result is False

    def test_build_messages_for_llm(self, mock_services):
        """Test building messages for LLM"""
        history = [
            {
                "role": "user",
                "content": "first message",
                "timestamp": "2023-01-01T12:00:00",
            },
            {
                "role": "assistant",
                "content": "first response",
                "timestamp": "2023-01-01T12:00:01",
            },
            {
                "role": "system",
                "content": "should be filtered",
                "timestamp": "2023-01-01T12:00:02",
            },
            {
                "role": "user",
                "content": "second message",
                "timestamp": "2023-01-01T12:00:03",
            },
        ]
        mock_services["storage"].get_conversation.return_value = history

        service = ChatService()
        messages = service._build_messages_for_llm("test-conv-id", "current query")

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "first message"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "first response"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "second message"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "current query"

    def test_build_messages_for_llm_exception(self, mock_services):
        """Test handling exception when building messages"""
        mock_services["storage"].get_conversation.side_effect = Exception(
            "Storage error"
        )

        service = ChatService()
        messages = service._build_messages_for_llm("test-conv-id", "current query")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "current query"

    @patch("db_chat.services.ChatService.get_db_schema")
    def test_handle_query_new_conversation(self, mock_get_schema, mock_services):
        """Test handling a query with a new conversation"""
        mock_get_schema.return_value = "Table schema"

        with patch("db_chat.services.ChatService.save_message", return_value=True):
            service = ChatService()

            def mock_generate_impl(user_query, schema, conversation_id):
                return {
                    "success": True,
                    "sql_query": "SELECT * FROM users",
                    "raw_result": "test result",
                }

            def mock_interpret_impl(
                user_query, sql_query, raw_result, schema, conversation_id
            ):
                return {
                    "reply": "Found 5 users.",
                    "conversation_id": conversation_id,
                    "sql_query": sql_query,
                    "raw_result": raw_result,
                }

            with patch.object(
                service, "_generate_and_execute_sql", side_effect=mock_generate_impl
            ) as mock_generate:
                with patch.object(
                    service, "_interpret_sql_results", side_effect=mock_interpret_impl
                ) as mock_interpret:
                    result = service.handle_query("Show me all users")

                    mock_services["storage"].create_conversation.assert_called_once()
                    mock_get_schema.assert_called_once()
                    mock_generate.assert_called_once()
                    assert result["reply"] == "Found 5 users."
                    assert result["conversation_id"] == "test-conversation-id"

    @patch("db_chat.services.ChatService.get_db_schema")
    def test_handle_query_existing_conversation(self, mock_get_schema, mock_services):
        """Test handling a query with an existing conversation"""
        mock_get_schema.return_value = "Table schema"
        mock_services["storage"].conversation_exists.return_value = True

        with patch("db_chat.services.ChatService.save_message", return_value=True):
            service = ChatService()

            def mock_generate_impl(user_query, schema, conversation_id):
                return {
                    "success": True,
                    "sql_query": "SELECT * FROM users WHERE id = 1",
                    "raw_result": "test result",
                }

            def mock_interpret_impl(
                user_query, sql_query, raw_result, schema, conversation_id
            ):
                return {
                    "reply": "Found user with id 1.",
                    "conversation_id": conversation_id,
                    "sql_query": sql_query,
                    "raw_result": raw_result,
                }

            with patch.object(
                service, "_generate_and_execute_sql", side_effect=mock_generate_impl
            ) as mock_generate:
                with patch.object(
                    service, "_interpret_sql_results", side_effect=mock_interpret_impl
                ) as mock_interpret:

                    result = service.handle_query("Find user with id 1", "existing-id")

                    mock_services[
                        "storage"
                    ].conversation_exists.assert_called_once_with("existing-id")
                    mock_services["storage"].update_expiry.assert_called_once_with(
                        "existing-id"
                    )
                    mock_get_schema.assert_called_once()
                    mock_generate.assert_called_once()
                    assert result["reply"] == "Found user with id 1."
                    assert result["conversation_id"] == "existing-id"

    @patch("db_chat.services.ChatService.get_db_schema")
    def test_handle_query_sql_execution_error(self, mock_get_schema, mock_services):
        """Test handling a query with SQL execution error"""
        mock_get_schema.return_value = "Table schema"

        with patch("db_chat.services.ChatService.save_message", return_value=True):
            service = ChatService()

            def mock_generate_impl(user_query, schema, conversation_id):
                return {
                    "success": False,
                    "sql_query": "SELECT * FROM nonexistent_table",
                    "error": "sql_execution_error",
                    "error_message": "Table doesn't exist",
                    "raw_result": "Error: Table nonexistent_table doesn't exist",
                }

            def mock_handle_error_impl(
                user_query, sql_query, error_message, conversation_id
            ):
                return {
                    "reply": "Table nonexistent_table doesn't exist.",
                    "conversation_id": conversation_id,
                    "error": "sql_execution_error",
                    "sql_query": sql_query,
                    "raw_result": error_message,
                }

            with patch.object(
                service, "_generate_and_execute_sql", side_effect=mock_generate_impl
            ) as mock_generate:
                with patch.object(
                    service, "handle_sql_error", side_effect=mock_handle_error_impl
                ) as mock_handle_error:
                    result = service.handle_query("Show me data from nonexistent table")

                    mock_get_schema.assert_called_once()
                    mock_generate.assert_called_once()
                    assert result["reply"] == "Table nonexistent_table doesn't exist."
                    assert result["error"] == "sql_execution_error"

    def test_validate_generated_sql_valid(self, mock_services):
        """Test validating a valid SQL query"""
        with patch("db_chat.services.is_valid_sql_structure") as mock_is_valid, patch(
            "db_chat.services.is_safe_sql"
        ) as mock_is_safe:

            mock_is_valid.return_value = True
            mock_is_safe.return_value = True

            service = ChatService()

            result = service._validate_generated_sql(
                "SELECT * FROM users", "test-conv-id"
            )

            assert result is None
            mock_is_valid.assert_called_once_with("SELECT * FROM users")
            mock_is_safe.assert_called_once_with(
                "SELECT * FROM users", ["users", "orders", "products"]
            )

    def test_validate_generated_sql_invalid_structure(self, mock_services):
        """Test validating SQL with invalid structure"""
        with patch("db_chat.services.is_valid_sql_structure") as mock_is_valid:

            mock_is_valid.return_value = False

            service = ChatService()

            result = service._validate_generated_sql("INVALID SQL", "test-conv-id")

            assert result is not None
            assert "error" in result
            assert result["error"] == "invalid_structure"
            mock_is_valid.assert_called_once_with("INVALID SQL")

    def test_validate_generated_sql_unsafe(self, mock_services):
        """Test validating unsafe SQL"""
        with patch("db_chat.services.is_valid_sql_structure") as mock_is_valid, patch(
            "db_chat.services.is_safe_sql"
        ) as mock_is_safe:

            mock_is_valid.return_value = True
            mock_is_safe.return_value = False

            service = ChatService()

            result = service._validate_generated_sql("DROP TABLE users", "test-conv-id")

            assert result is not None
            assert "error" in result
            assert result["error"] == "security_violation"
            mock_is_valid.assert_called_once_with("DROP TABLE users")
            mock_is_safe.assert_called_once_with(
                "DROP TABLE users", ["users", "orders", "products"]
            )

    def test_get_or_create_conversation_new(self, mock_services):
        service = ChatService()
        mock_services["storage"].conversation_exists.return_value = False
        cid = service._get_or_create_conversation(None)
        assert cid == "test-conversation-id"

    def test_get_or_create_conversation_existing(self, mock_services):
        service = ChatService()
        mock_services["storage"].conversation_exists.return_value = True
        cid = service._get_or_create_conversation("existing-id")
        assert cid == "existing-id"

    def test_get_or_create_conversation_error(self, mock_services):
        service = ChatService()
        mock_services["storage"].conversation_exists.side_effect = Exception("fail")
        cid = service._get_or_create_conversation("bad-id")
        assert cid is None

    def test_get_schema_or_error_success(self, mock_services):
        service = ChatService()
        mock_services["handler"].get_schema.return_value = "schema"
        result = service._get_schema_or_error("cid")
        assert result == "schema"

    def test_get_schema_or_error_error(self, mock_services):
        service = ChatService()
        mock_services["handler"].get_schema.return_value = "Error: fail"
        result = service._get_schema_or_error("cid")
        assert result.startswith("Sorry, I encountered an issue")

    def test_conversation_and_schema_success(self, mock_services):
        service = ChatService()
        mock_services["storage"].conversation_exists.return_value = True
        mock_services["handler"].get_schema.return_value = "schema"
        cid, schema, error = service._conversation_and_schema("query", "cid")
        assert cid == "cid"
        assert schema == "schema"
        assert error is None

    def test_conversation_and_schema_error(self, mock_services):
        service = ChatService()
        mock_services["storage"].conversation_exists.return_value = True
        mock_services["handler"].get_schema.return_value = "Error: fail"
        cid, schema, error = service._conversation_and_schema("query", "cid")
        assert error is not None
        assert error["error"] == "schema_error"


@pytest.mark.asyncio
async def test_handle_query_stream_success(mock_services):
    service = ChatService()
    with patch.object(
        service, "_conversation_and_schema", return_value=("cid", "schema", None)
    ), patch.object(
        service,
        "_generate_and_execute_sql",
        return_value={"sql_query": "SELECT", "raw_result": "result"},
    ), patch.object(
        service.llm_adapter,
        "stream_text",
        return_value=async_iter(["Hello ", "world!"]),
    ):
        results = [msg async for msg in service.handle_query_stream("query")]
        assert any(m["type"] == "llm_token" for m in results)
        assert any(m["type"] == "llm_stream_end" for m in results)


@pytest.mark.asyncio
async def test_handle_query_stream_error_streaming(mock_services):
    service = ChatService()
    with patch.object(
        service, "_conversation_and_schema", return_value=("cid", "schema", None)
    ), patch.object(
        service,
        "_generate_and_execute_sql",
        return_value={"sql_query": "SELECT", "raw_result": "Error: fail"},
    ), patch.object(
        service,
        "_stream_error_explanation",
        return_value=async_iter(
            [
                {"type": "llm_token", "token": "Error", "conversation_id": "cid"},
                {
                    "type": "llm_stream_end",
                    "message": "Error",
                    "conversation_id": "cid",
                },
            ]
        ),
    ):
        results = [msg async for msg in service.handle_query_stream("query")]
        assert any(m["type"] == "llm_token" for m in results)
        assert any(m["type"] == "llm_stream_end" for m in results)


@pytest.mark.asyncio
async def test_stream_error_explanation(mock_services):
    service = ChatService()
    with patch.object(
        service.llm_adapter,
        "stream_text",
        return_value=async_iter(["E", "r", "r", "o", "r"]),
    ), patch.object(service, "save_message", return_value=True):
        chunks = [
            msg
            async for msg in service._stream_error_explanation(
                "q", "sql", "fail", "schema", "cid"
            )
        ]
        assert any(m["type"] == "llm_token" for m in chunks)
        assert any(m["type"] == "llm_stream_end" for m in chunks)


def async_iter(items):
    async def _aiter():
        for i in items:
            yield i

    return _aiter()
