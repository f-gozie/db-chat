"""Helper functions for the db_chat application."""

import logging

from .constants import DatabaseDialects

logger = logging.getLogger(__name__)


def get_dialect_syntax_example(db_dialect: str) -> str:
    """Returns a dialect-specific syntax example string.

    Args:
        db_dialect (str): The database dialect (e.g., DatabaseDialects.POSTGRESQL).

    Returns:
        str: A string containing a syntax example relevant to the dialect, or an empty string.
    """
    if db_dialect == DatabaseDialects.POSTGRESQL:
        return (
            "For example, use `CURRENT_DATE - INTERVAL '1 year'` for date calculations."
        )
    elif db_dialect == DatabaseDialects.MYSQL:
        return "For example, use `DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)` for date calculations."
    elif db_dialect == DatabaseDialects.SQLITE:
        return "For example, use `date('now', '-1 year')` for date calculations."
    else:
        logger.warning(f"No specific syntax example defined for dialect: {db_dialect}")
        return ""
