from unittest.mock import MagicMock

import django
import pytest
from django.conf import settings
from django.db.models.fields import related as dj_related

from db_chat.model_registry import ModelRegistry, diagnostic_info, get_registry

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

    def test_get_table_schema_returns_none_for_missing(self):
        reg = ModelRegistry()
        reg.initialized = True
        reg.models_info = {}
        assert reg.get_table_schema("not_a_table") is None

    def test_get_table_schema_formats_schema(self):
        reg = ModelRegistry()
        reg.initialized = True
        reg.models_info = {
            "my_table": {
                "table_name": "my_table",
                "fields": {
                    "id": {
                        "field_type": "AutoField",
                        "is_primary_key": True,
                        "is_unique": True,
                        "is_null": False,
                        "default": None,
                        "choices": None,
                        "related_model": None,
                        "column_name": "id",
                        "help_text": None,
                        "max_length": None,
                    },
                    "name": {
                        "field_type": "CharField",
                        "is_primary_key": False,
                        "is_unique": False,
                        "is_null": True,
                        "default": None,
                        "choices": [(1, "A"), (2, "B")],
                        "related_model": None,
                        "column_name": "name",
                        "help_text": "help",
                        "max_length": 100,
                    },
                },
                "constraints": [{"name": "unique_name", "type": "UniqueConstraint"}],
                "app_label": "app",
                "model_name": "MyModel",
            }
        }
        schema = reg.get_table_schema("my_table")
        assert "Table: my_table" in schema
        assert "id (id): AutoField PRIMARY KEY UNIQUE NOT NULL" in schema

    def test_get_registry_returns_singleton(self):
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_diagnostic_info(self):
        reg = get_registry()
        reg.models_info = {
            "t": {"fields": {"id": {}, "name": {}}, "app_label": "a", "model_name": "b"}
        }
        reg.initialized = True
        info = diagnostic_info()
        assert "t" in info["tables"]
        assert "a.b" in info["model_paths"]

    def test_extract_fields_info_and_constraints(self):
        class DummyField:
            def __init__(
                self,
                name,
                internal_type,
                pk=False,
                unique=False,
                null=False,
                default=None,
                choices=None,
                related_model=None,
                column=None,
                help_text=None,
                max_length=None,
            ):
                self.name = name
                self.get_internal_type = lambda: internal_type
                self.primary_key = pk
                self.unique = unique
                self.null = null
                self.default = default
                self.choices = choices
                self.related_model = related_model
                self.column = column or name
                self.help_text = help_text
                self.max_length = max_length

        class DummyConstraint:
            def __init__(self, name):
                self.name = name
                self.__class__.__name__ = "UniqueConstraint"

        class DummyMeta:
            def __init__(self):
                self.fields = [
                    DummyField("id", "AutoField", pk=True, unique=True),
                    DummyField(
                        "name",
                        "CharField",
                        max_length=50,
                        help_text="desc",
                        choices=[(1, "A")],
                    ),
                ]
                self.many_to_many = [
                    DummyField("tags", "ManyToManyField", related_model="app.Tag")
                ]
                self.constraints = [DummyConstraint("uniq")]
                self.db_table = "dummy"
                self.abstract = False
                self.model_name = "dummy"
                self.app_label = "app"

        class DummyModel:
            _meta = DummyMeta()

        reg = ModelRegistry()
        fields = reg._extract_fields_info(DummyModel)
        assert "id" in fields and "name" in fields and "tags" in fields
        constraints = reg._extract_constraints(DummyModel)
        assert constraints[0]["name"] == "uniq"
        assert constraints[0]["type"] == "UniqueConstraint"

    def test_get_choices_and_related_model(self):
        reg = ModelRegistry()

        class DummyRelated:
            _meta = type("Meta", (), {"app_label": "a", "model_name": "b"})

        class DummyField:
            def __init__(self, choices=None, related_model=None):
                self.choices = choices
                self.related_model = related_model

        import django.db.models.fields.related as dj_related

        orig_isinstance = isinstance

        def fake_isinstance(obj, cls):
            if cls is dj_related.RelatedField and isinstance(obj, DummyField):
                return True
            return orig_isinstance(obj, cls)

        import builtins

        builtins.isinstance, old_isinstance = fake_isinstance, builtins.isinstance
        try:
            field = DummyField(choices=[(1, "A")], related_model=DummyRelated)
            choices, related_model = reg._get_choices_and_related_model(field)
            assert choices == [(1, "A")]
            assert related_model == DummyRelated
        finally:
            builtins.isinstance = old_isinstance
