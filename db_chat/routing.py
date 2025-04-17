from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r"^ws/db_chat/query/$", consumers.ChatConsumer.as_asgi()),
]
