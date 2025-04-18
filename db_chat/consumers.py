import json
import logging

from asgiref.sync import sync_to_async
from channels.exceptions import DenyConnection
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

from .services import ChatService

logger = logging.getLogger(__name__)


chat_service = ChatService()

chat_service_handle_query = sync_to_async(
    chat_service.handle_query, thread_sensitive=True
)


class ChatConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for handling real-time chat interactions.
    Uses sync_to_async to interact with the synchronous ChatService.
    """

    async def connect(self):
        """
        Called when the websocket is trying to connect.
        Accepts the connection only if the WebSocket feature is enabled.
        """
        if not getattr(settings, "DB_CHAT_ENABLE_WEBSOCKETS", False):
            raise DenyConnection("WebSocket feature not enabled.")

        # TODO: Add authentication/authorization logic if needed

        await self.accept()
        await self.send_status("Connection established. Waiting for query...")

    async def disconnect(self, close_code):
        """
        Called when the WebSocket closes for any reason.

        Args:
            close_code (int): The WebSocket close code.
        """
        logger.info(f"WebSocket connection closed with code: {close_code}")

    async def receive(self, text_data):
        """
        Called when a message is received from the WebSocket.
        Parses the message and delegates to the ChatService.
        """
        try:
            data = json.loads(text_data)
            query = data.get("query")
            conversation_id = data.get("conversation_id")
            stream = data.get("stream", False)

            if not query:
                await self.send_error("Missing 'query' in message.")
                return

            if stream:
                await self.handle_streaming_query(query, conversation_id)
            else:
                await self.send_status(f"Processing query: '{query[:50]}...'")
                try:
                    response_data = await chat_service_handle_query(
                        user_query=query, conversation_id=conversation_id
                    )
                    await self.send(
                        text_data=json.dumps({"type": "chat_response", **response_data})
                    )
                except Exception as e:
                    logger.exception(
                        f"Error during handle_query execution in ChatConsumer: {e}"
                    )
                    await self.send_error(
                        f"An unexpected error occurred while processing your request."
                    )

        except json.JSONDecodeError:
            logger.warning("Invalid JSON received in ChatConsumer.")
            await self.send_error("Invalid JSON received.")
        except DenyConnection as e:
            logger.warning(f"Connection denied in ChatConsumer: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unhandled exception in ChatConsumer receive: {e}")
            await self.send_error("An unexpected server error occurred.")

    async def send_error(self, message):
        """Helper method to send an error message back to the client."""
        await self.send(text_data=json.dumps({"type": "error", "message": message}))

    async def send_status(self, message):
        """Helper method to send a status update message back to the client."""
        await self.send(text_data=json.dumps({"type": "status", "message": message}))

    async def handle_streaming_query(self, query, conversation_id):
        """
        Handles streaming LLM responses over the websocket.
        """
        await self.send_status(f"Streaming response for: '{query[:50]}...'")
        try:
            async for chunk in ChatService().handle_query_stream(
                query, conversation_id
            ):
                await self.send(text_data=json.dumps(chunk))
        except Exception as e:
            logger.exception(f"Error during streaming query: {e}")
            await self.send_error(f"Streaming error: {str(e)}")
