from django.urls import path

from db_chat import views

urlpatterns = [
    path("chat/", views.chat_view, name="chat"),
    path(
        "model-registry-diagnostic/",
        views.model_registry_diagnostic,
        name="model_registry_diagnostic",
    ),
]
