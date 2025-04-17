import json
from unittest.mock import AsyncMock, patch

import pytest
from channels.exceptions import DenyConnection
from django.conf import settings

with patch("db_chat.services.ChatService._initialize_llm_adapter"):
    from db_chat.consumers import ChatConsumer


@pytest.mark.asyncio
async def test_connect_when_enabled():
    """Test that connect accepts connection when websockets are enabled"""
    settings.DB_CHAT_ENABLE_WEBSOCKETS = True
    consumer = ChatConsumer()
    consumer.accept = AsyncMock()
    consumer.send_status = AsyncMock()

    await consumer.connect()
    consumer.accept.assert_called_once()
    consumer.send_status.assert_called_once_with(
        "Connection established. Waiting for query..."
    )


@pytest.mark.asyncio
async def test_connect_when_disabled():
    """Test that connect rejects connection when websockets are disabled"""
    settings.DB_CHAT_ENABLE_WEBSOCKETS = False
    consumer = ChatConsumer()

    with pytest.raises(DenyConnection):
        await consumer.connect()


@pytest.mark.asyncio
async def test_disconnect():
    """Test disconnect method logs the close code"""
    consumer = ChatConsumer()

    with patch("db_chat.consumers.logger") as mock_logger:
        await consumer.disconnect(1000)
        mock_logger.info.assert_called_once()


@pytest.mark.asyncio
async def test_receive_valid_query():
    """Test processing a valid query"""
    consumer = ChatConsumer()
    consumer.send_status = AsyncMock()
    consumer.send = AsyncMock()

    test_query = "What tables are available?"
    test_conversation_id = "test-convo-123"
    test_response = {
        "reply": "Test reply",
        "conversation_id": test_conversation_id,
        "sql": "SELECT test",
    }

    with patch(
        "db_chat.consumers.chat_service_handle_query", new_callable=AsyncMock
    ) as mock_handle:
        mock_handle.return_value = test_response

        await consumer.receive(
            text_data=json.dumps(
                {"query": test_query, "conversation_id": test_conversation_id}
            )
        )

        consumer.send_status.assert_called_once()
        mock_handle.assert_called_once_with(
            user_query=test_query, conversation_id=test_conversation_id
        )
        consumer.send.assert_called_once_with(
            text_data=json.dumps({"type": "chat_response", **test_response})
        )


@pytest.mark.asyncio
async def test_receive_invalid_json():
    """Test handling invalid JSON"""
    consumer = ChatConsumer()
    consumer.send_error = AsyncMock()

    await consumer.receive(text_data="not valid json")
    consumer.send_error.assert_called_once_with("Invalid JSON received.")


@pytest.mark.asyncio
async def test_receive_missing_query():
    """Test handling message without query field"""
    consumer = ChatConsumer()
    consumer.send_error = AsyncMock()

    await consumer.receive(text_data=json.dumps({"conversation_id": "test-convo-123"}))

    consumer.send_error.assert_called_once_with("Missing 'query' in message.")


@pytest.mark.asyncio
async def test_receive_chat_service_error():
    """Test handling error from chat service"""
    consumer = ChatConsumer()
    consumer.send_status = AsyncMock()
    consumer.send_error = AsyncMock()

    with patch(
        "db_chat.consumers.chat_service_handle_query", new_callable=AsyncMock
    ) as mock_handle:
        mock_handle.side_effect = Exception("Test error")

        await consumer.receive(text_data=json.dumps({"query": "test query"}))

        consumer.send_error.assert_called_once()


@pytest.mark.asyncio
async def test_receive_general_exception():
    """Test handling unspecified exception"""
    consumer = ChatConsumer()
    consumer.send_error = AsyncMock()

    with patch("json.loads", side_effect=Exception("General error")):
        await consumer.receive(text_data="something")
        consumer.send_error.assert_called_once_with(
            "An unexpected server error occurred."
        )


@pytest.mark.asyncio
async def test_send_error():
    """Test send_error helper method"""
    consumer = ChatConsumer()
    consumer.send = AsyncMock()
    error_message = "Test error message"

    await consumer.send_error(error_message)

    consumer.send.assert_called_once_with(
        text_data=json.dumps({"type": "error", "message": error_message})
    )


@pytest.mark.asyncio
async def test_send_status():
    """Test send_status helper method"""
    consumer = ChatConsumer()
    consumer.send = AsyncMock()
    status_message = "Test status message"

    await consumer.send_status(status_message)

    consumer.send.assert_called_once_with(
        text_data=json.dumps({"type": "status", "message": status_message})
    )
