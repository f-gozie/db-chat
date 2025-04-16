"""App configuration for the database chat application."""

import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)

class DbChatConfig(AppConfig):
    """Django app configuration for the database chat application."""
    
    default_auto_field = "django.db.models.BigAutoField"
    name = "db_chat"
    verbose_name = "Database Chat"
    
    def ready(self):
        """Initialize components when Django app is ready."""
        try:
            from .model_registry import get_registry
            registry = get_registry()
            registry.initialize()
            logger.info(f"Model registry initialized with {len(registry.get_all_tables())} tables")
        except Exception as e:
            logger.warning(f"Error initializing model registry: {e}")
            logger.warning("Model registry will be initialized on first request")