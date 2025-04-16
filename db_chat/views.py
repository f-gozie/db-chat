"""View handlers for the database chat API endpoints."""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
import logging

from .services import ChatService
from .model_registry import diagnostic_info

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
def chat_view(request):
    """Endpoint for chat interactions that translates natural language to SQL queries.
    
    Request format:
    {
        "query": "the user's question",
        "conversation_id": "optional - UUID of an existing conversation"
    }
    
    Response format:
    {
        "reply": "response from the assistant",
        "sql_query": "the generated SQL query",
        "raw_result": "raw SQL result",
        "conversation_id": "UUID of the conversation"
    }
    """
    try:
        data = json.loads(request.body)
        user_query = data.get("query")
        conversation_id = data.get("conversation_id")

        if not user_query:
            return JsonResponse({"error": "Missing 'query' in request body"}, status=400)

        chat_service = ChatService()
        response = chat_service.handle_query(user_query, conversation_id)
        
        logger.info(f"Processed query for conversation ID {response.get('conversation_id')}")
        return JsonResponse(response)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        logger.error(f"Error in chat_view: {e}")
        return JsonResponse({"error": "An internal server error occurred"}, status=500)

@csrf_exempt
def model_registry_diagnostic(request):
    """Endpoint providing diagnostic information about the model registry.
    
    Returns details about detected models, their relationships, and fields with choices.
    """
    return JsonResponse(diagnostic_info())
