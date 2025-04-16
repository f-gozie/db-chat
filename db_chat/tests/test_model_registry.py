from unittest.mock import MagicMock

import django
import pytest
from django.conf import settings

from db_chat.model_registry import ModelRegistry

# Ensure Django settings are configured for these tests
if not settings.configured:
    settings.configure(ALLOWED_MODELS=["app.Model"], INSTALLED_APPS=[], DATABASES={})
    django.setup()


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    class DummySettings:
        ALLOWED_MODELS = ["app.Model"]

    monkeypatch.setattr("db_chat.model_registry.settings", DummySettings)
    yield


@pytest.fixture(autouse=True)
def patch_apps(monkeypatch):
    monkeypatch.setattr("db_chat.model_registry.apps", MagicMock())
    yield


class TestModelRegistry:
    def test_initialize_with_valid_models(self, monkeypatch):
        mock_model = MagicMock()
        mock_model._meta.db_table = "table"
        mock_model._meta.abstract = False
        mock_model._meta.model_name = "model"
        mock_model._meta.app_label = "app"
        mock_model._meta.fields = []
        mock_model._meta.many_to_many = []
        monkeypatch.setattr(
            "db_chat.model_registry.apps.get_model", lambda app, model: mock_model
        )
        reg = ModelRegistry()
        reg.initialize()
        assert reg.initialized
        assert "table" in reg.models_info

    def test_initialize_no_models(self, monkeypatch):
        class DummySettings:
            ALLOWED_MODELS = []

        monkeypatch.setattr("db_chat.model_registry.settings", DummySettings)
        reg = ModelRegistry()
        reg.initialize()
        assert reg.initialized
        assert reg.models_info == {}

    def test_initialize_invalid_spec(self, monkeypatch):
        class DummySettings:
            ALLOWED_MODELS = ["bad"]

        monkeypatch.setattr("db_chat.model_registry.settings", DummySettings)
        monkeypatch.setattr(
            "db_chat.model_registry.apps.get_model",
            lambda app, model: (_ for _ in ()).throw(Exception("fail")),
        )
        reg = ModelRegistry()
        reg.initialize()
        assert reg.initialized

    def test_get_all_tables_and_fields(self):
        reg = ModelRegistry()
        reg.models_info = {"table": {"fields": {"id": {}, "name": {}}}}
        reg.initialized = True
        assert reg.get_all_tables() == ["table"]
        assert reg.get_tables_and_fields() == {"table": ["id", "name"]}
