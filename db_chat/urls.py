from django.urls import path

from .views import chat_view, model_registry_diagnostic

urlpatterns = [
    path("query/", chat_view, name="chat_view"),
    path("schema-info/", model_registry_diagnostic, name="model_registry_diagnostic"),
]
