import logging
import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.conf import settings
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "db_chat_project.settings")
django_asgi_app = get_asgi_application()

websocket_routes = []

logger = logging.getLogger(__name__)

if getattr(settings, "DB_CHAT_ENABLE_WEBSOCKETS", False):
    try:
        from db_chat.routing import websocket_urlpatterns

        websocket_routes.extend(websocket_urlpatterns)
        logger.info("db_chat WebSocket routes enabled and loaded.")
    except ImportError:
        logger.warning(
            "DB_CHAT_ENABLE_WEBSOCKETS is True, but db_chat.routing could not be imported. Skipping WebSocket routes."
        )
    except AttributeError:
        logger.warning(
            "DB_CHAT_ENABLE_WEBSOCKETS is True, but websocket_urlpatterns not found in db_chat.routing. Skipping WebSocket routes."
        )
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred while trying to load db_chat WebSocket routes: {e}"
        )


application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AuthMiddlewareStack(URLRouter(websocket_routes))
        if websocket_routes
        else None,
        # Add other protocols here if needed
    }
)
