"""Module to extract schema information from Django models for the database chat application."""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from django.apps import apps
from django.conf import settings
from django.db import models
from django.db.models.fields import related

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry that extracts and provides metadata about Django models."""

    def __init__(self):
        self.models_info = {}
        self.initialized = False
        self.dependency_graph = defaultdict(set)

        # Configuration flags (can be overridden via Django settings or initialize params)
        self.include_fk = True
        self.include_m2m = True
        self.include_all_models = False
        self.include_apps: List[str] = []

    def initialize(
        self,
        model_specs=None,
        *,
        include_fk: Optional[bool] = None,
        include_m2m: Optional[bool] = None,
        include_all_models: Optional[bool] = None,
        include_apps: Optional[List[str]] = None,
    ):
        """Load and process specified models and their dependencies.

        Args:
            model_specs: List of model specifications in "app_label.ModelName" format.
                         If None, try to get from settings.ALLOWED_MODELS.
        """
        if self.initialized:
            return

        logger.info("Initializing model registry...")

        # Merge supplied flags with settings‑based configuration
        settings_opts = getattr(settings, "MODEL_REGISTRY_OPTIONS", {})

        self.include_fk = (
            include_fk
            if include_fk is not None
            else settings_opts.get("include_fk", True)
        )
        self.include_m2m = (
            include_m2m
            if include_m2m is not None
            else settings_opts.get("include_m2m", True)
        )
        self.include_all_models = (
            include_all_models
            if include_all_models is not None
            else settings_opts.get("include_all_models", False)
        )
        self.include_apps = (
            include_apps
            if include_apps is not None
            else settings_opts.get("include_apps", [])
        )

        # Decide which models to load:
        # 1. If include_all_models -> load everything
        # 2. Else if include_apps -> load all models from specified apps
        # 3. Else fall back to model_specs (or ALLOWED_MODELS setting)

        if self.include_all_models:
            all_models = apps.get_models()
            for m in all_models:
                self._process_model_and_dependencies(m)
            self.initialized = True
            logger.info(
                f"Model registry initialized (ALL MODELS) with {len(self.models_info)} tables"
            )
            return

        if self.include_apps:
            for app_label in self.include_apps:
                try:
                    app_config = apps.get_app_config(app_label)
                    for m in app_config.get_models():
                        self._process_model_and_dependencies(m)
                except LookupError:
                    logger.warning(f"App '{app_label}' not found in INSTALLED_APPS.")

            # If no explicit model_specs provided, we're done
            if model_specs is None:
                self.initialized = True
                logger.info(
                    f"Model registry initialized (apps={self.include_apps}) with {len(self.models_info)} tables"
                )
                return

        # Get model specifications from settings if not provided
        if model_specs is None:
            model_specs = getattr(settings, "ALLOWED_MODELS", [])

        if not model_specs:
            logger.warning(
                "No models specified. Configure ALLOWED_MODELS in settings or use include_all_models/include_apps flags."
            )
            self.initialized = True
            return

        # Process each model specification
        for model_spec in model_specs:
            try:
                if "." in model_spec:
                    app_label, model_name = model_spec.split(".")
                    model = apps.get_model(app_label, model_name)
                    self._process_model_and_dependencies(model)
                else:
                    logger.warning(
                        f"Model spec '{model_spec}' doesn't follow 'app_label.ModelName' format. Skipping."
                    )
            except Exception as e:
                logger.warning(f"Error processing model spec '{model_spec}': {str(e)}")

        self.initialized = True
        logger.info(f"Model registry initialized with {len(self.models_info)} models")

    def _process_model_and_dependencies(self, model, visited=None):
        """Process a model and its dependencies recursively."""
        if visited is None:
            visited = set()

        # Skip if already processed or abstract
        if model._meta.db_table in self.models_info or model._meta.abstract:
            return

        if model in visited:
            return
        visited.add(model)

        # Process this model
        model_name = model._meta.model_name
        app_label = model._meta.app_label
        table_name = model._meta.db_table

        logger.debug(f"Processing model {app_label}.{model_name} (table: {table_name})")

        # First collect dependencies to build the dependency graph
        dependencies: List[models.Model] = []

        # Follow ForeignKey / OneToOne relationships
        if self.include_fk:
            for field in model._meta.fields:
                if isinstance(field, (related.ForeignKey, related.OneToOneField)):
                    related_model = field.related_model
                    if related_model and not related_model._meta.abstract:
                        dependencies.append(related_model)
                        self.dependency_graph[model].add(related_model)

        # Follow Many‑to‑Many relationships (including explicit through tables)
        if self.include_m2m:
            for field in model._meta.many_to_many:
                related_model = field.related_model
                if related_model and not related_model._meta.abstract:
                    dependencies.append(related_model)
                    self.dependency_graph[model].add(related_model)

                # Include the through/intermediate model for the M2M relation
                through_model = getattr(field.remote_field, "through", None)
                if through_model and not through_model._meta.abstract:
                    dependencies.append(through_model)
                    self.dependency_graph[model].add(through_model)

        # Store model info
        model_info = {
            "model": model,
            "app_label": app_label,
            "model_name": model_name,
            "table_name": table_name,
            "fields": self._extract_fields_info(model),
            "constraints": self._extract_constraints(model),
        }

        # Add to registry by table name for easy lookup
        self.models_info[table_name] = model_info

        # Process dependencies
        for dep_model in dependencies:
            self._process_model_and_dependencies(dep_model, visited)

    def _extract_fields_info(self, model) -> Dict[str, Dict[str, Any]]:
        """Extract detailed information about model fields."""
        fields_info = {}

        for field in model._meta.fields:
            field_name = field.name
            field_info = {
                "field_name": field_name,
                "field_type": field.get_internal_type(),
                "is_primary_key": field.primary_key,
                "is_unique": field.unique,
                "is_null": field.null,
                "default": field.default
                if field.default != models.fields.NOT_PROVIDED
                else None,
                "choices": self._get_choices(field),
                "related_model": self._get_related_model(field),
                "column_name": field.column,
                "help_text": str(field.help_text) if field.help_text else None,
                "max_length": getattr(field, "max_length", None),
            }

            fields_info[field_name] = field_info

        # Also include many-to-many fields
        for field in model._meta.many_to_many:
            field_name = field.name
            field_info = {
                "field_name": field_name,
                "field_type": field.get_internal_type(),
                "is_primary_key": False,
                "is_unique": False,
                "is_null": True,  # M2M fields are effectively nullable
                "default": None,
                "choices": None,
                "related_model": self._get_related_model(field),
                "column_name": field.column,  # This will be the name in the intermediary table
                "help_text": str(field.help_text) if field.help_text else None,
                "is_many_to_many": True,
            }

            fields_info[field_name] = field_info

        return fields_info

    def _get_choices(self, field) -> Optional[List[Tuple[Any, str]]]:
        """Extract choices from field if present."""
        if hasattr(field, "choices") and field.choices:
            # Return the actual field choices
            return field.choices
        return None

    def _get_related_model(self, field) -> Optional[str]:
        """Get related model for relational fields."""
        if isinstance(field, related.RelatedField):
            related_model = field.related_model
            if related_model:
                return (
                    f"{related_model._meta.app_label}.{related_model._meta.model_name}"
                )
        return None

    def _get_choices_and_related_model(self, field):
        """Return a tuple (choices, related_model) for the given field."""
        choices = self._get_choices(field)
        related_model = None
        if isinstance(field, related.RelatedField):
            related_model = getattr(field, "related_model", None)
        return (choices, related_model)

    def _extract_constraints(self, model) -> List[Dict[str, Any]]:
        """Extract model constraints."""
        constraints = []

        if hasattr(model._meta, "constraints"):
            for constraint in model._meta.constraints:
                constraint_info = {
                    "name": constraint.name,
                    "type": constraint.__class__.__name__,
                }
                constraints.append(constraint_info)

        return constraints

    def get_table_schema(self, table_name: str) -> Optional[str]:
        """Get formatted schema information for a specific table."""
        if not self.initialized:
            self.initialize()

        model_info = self.models_info.get(table_name)
        if not model_info:
            return None

        schema_lines = [
            f"Table: {table_name} (Django Model: {model_info['app_label']}.{model_info['model_name']})\n"
        ]

        for field_name, field_info in model_info["fields"].items():
            field_type = field_info["field_type"]
            db_column_name = field_info["column_name"]

            # Add specific PostgreSQL querying info for JSONField
            json_info = ""
            if field_type == "JSONField":
                json_info = (
                    f" (JSON object. Query keys in PostgreSQL: "
                    f"(\"{db_column_name}\" ->> 'key_name')::datatype, "
                    f"check key exists: \"{db_column_name}\" ? 'key_name')"
                )

            constraints = []
            if field_info["is_primary_key"]:
                constraints.append("PRIMARY KEY")
            if field_info["is_unique"]:
                constraints.append("UNIQUE")
            if not field_info["is_null"]:
                constraints.append("NOT NULL")

            default_val = field_info["default"]
            if (
                default_val is not None
                and default_val != models.fields.NOT_PROVIDED
                and default_val != ""
            ):
                default_str = (
                    json.dumps(default_val)
                    if isinstance(default_val, dict)
                    else repr(default_val)
                )
                constraints.append(f"DEFAULT {default_str}")

            if field_info["related_model"]:
                # Basic assumption: related model string format is AppLabel.ModelName
                # This might need adjustment based on actual _get_related_model output
                related_parts = field_info["related_model"].split(".")
                if len(related_parts) == 2:
                    try:
                        related_model_ref = apps.get_model(
                            related_parts[0], related_parts[1]
                        )
                        related_table = related_model_ref._meta.db_table
                        related_pk = related_model_ref._meta.pk.name
                        constraints.append(f"REFERENCES {related_table}({related_pk})")
                    except LookupError:
                        constraints.append(
                            f"REFERENCES {field_info['related_model']} (unknown PK)"
                        )  # Fallback
                else:
                    constraints.append(
                        f"REFERENCES {field_info['related_model']} (unknown PK)"
                    )

            constraints_str = " ".join(constraints)

            choices_str = ""
            if field_info["choices"]:
                # Build CHOICES with both value and display label for clarity
                choice_items = []
                for val, name in field_info["choices"]:
                    val_repr = repr(val)
                    name_repr = repr(name)
                    choice_items.append(f"{val_repr}: {name_repr}")
                choices_str = f" CHOICES: ({', '.join(choice_items)})"

            field_def = f"  - {field_name} ({db_column_name}): {field_type}{json_info} {constraints_str}{choices_str}"
            schema_lines.append(field_def)

        return "\n".join(schema_lines)

    def get_all_tables(self) -> List[str]:
        """Get a list of all available table names."""
        if not self.initialized:
            self.initialize()

        return list(self.models_info.keys())

    def get_tables_and_fields(self) -> Dict[str, List[str]]:
        """Get a dictionary of all tables and their fields."""
        if not self.initialized:
            self.initialize()

        result = {}
        for table_name, model_info in self.models_info.items():
            result[table_name] = list(model_info["fields"].keys())

        return result


# Singleton instance
registry = ModelRegistry()


def get_registry():
    """Get the model registry instance."""
    return registry


def diagnostic_info():
    """Return diagnostic information about the model registry."""
    registry = get_registry()
    if not registry.initialized:
        registry.initialize()

    info = {
        "initialized": registry.initialized,
        "model_count": len(registry.models_info),
        "tables": list(registry.models_info.keys()),
        "model_paths": [
            f"{info['app_label']}.{info['model_name']}"
            for info in registry.models_info.values()
        ],
        "choice_fields": {},
    }

    # Find fields with choices
    for table, model_info in registry.models_info.items():
        for field_name, field_info in model_info["fields"].items():
            if field_info.get("choices"):
                if table not in info["choice_fields"]:
                    info["choice_fields"][table] = {}

                choice_values = [str(choice[0]) for choice in field_info["choices"]]
                info["choice_fields"][table][field_name] = choice_values

    # Build dependency graph visualization
    dependency_viz = {}
    for model, deps in registry.dependency_graph.items():
        model_path = f"{model._meta.app_label}.{model._meta.model_name}"
        dependency_viz[model_path] = [
            f"{d._meta.app_label}.{d._meta.model_name}" for d in deps
        ]

    info["dependencies"] = dependency_viz

    return info
