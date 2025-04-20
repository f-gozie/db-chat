"""Component interfaces and default implementations for pluggable behaviour.

This module centralises *optional* building blocks that can be swapped via
Django settings.  Everything here is intentionally lightweight so the app
continues to work out‑of‑the‑box without further configuration.

The goal is to let advanced users plug in custom logic (e.g. a powerful LLM‑
based intent classifier) by pointing dotted paths in settings, while casual
installations rely on the defaults defined below.
"""
from __future__ import annotations

import importlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: dotted‑path loader
# ---------------------------------------------------------------------------


def import_from_dotted_path(dotted_path: str):
    """Dynamically import a class/function from a dotted path.

    Parameters
    ----------
    dotted_path: str
        e.g. ``"db_chat.components.DefaultIntentClassifier"``

    Returns
    -------
    The imported attribute.
    """

    module_path, _, attr = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(
            f"Invalid dotted path '{dotted_path}'. It should look like 'package.module.ClassName'."
        )
    module = importlib.import_module(module_path)
    return getattr(module, attr)


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class BaseIntentClassifier(ABC):
    """Interface for intent / clarification detection.

    The classifier inspects a user query and returns a result dict.
    At minimum it should contain ``requires_clarification`` (bool) and
    optionally ``clarification_prompt`` when clarification is needed.
    """

    @abstractmethod
    def classify(self, user_query: str) -> Dict[str, Any]:
        """Analyse *user_query* and decide whether clarification is required.

        Returns
        -------
        Dict[str, Any]
            Example::

                {
                  "requires_clarification": False,
                  "clarification_prompt": "",
                  "intent": "aggregation"  # optional
                }
        """


class BasePostProcessor(ABC):
    """Interface for transforming raw DB results into user‑friendly replies."""

    @abstractmethod
    def process(
        self,
        user_query: str,
        sql_query: str,
        raw_result: Any,
        technical: bool = False,
    ) -> str:
        """Return a natural‑language answer derived from *raw_result*."""


# ---------------------------------------------------------------------------
# Default implementations
# ---------------------------------------------------------------------------


class DefaultIntentClassifier(BaseIntentClassifier):
    """A heuristic intent classifier.

    – If the query looks too vague (e.g. contains *what can you do* or
      is shorter than 4 words), we mark it for clarification.
    – Otherwise, no clarification is necessary.
    """

    _VAGUE_PATTERNS = [
        re.compile(r"\bwhat can you do\b", re.I),
        re.compile(r"^\s*help\s*$", re.I),
    ]

    def classify(self, user_query: str) -> Dict[str, Any]:
        query = user_query.strip()
        is_vague = any(p.search(query) for p in self._VAGUE_PATTERNS)

        if is_vague:
            return {
                "requires_clarification": True,
                "clarification_prompt": (
                    "Could you provide a bit more detail about what you're looking for?"
                ),
                "intent": None,
            }
        return {
            "requires_clarification": False,
            "clarification_prompt": "",
            "intent": None,
        }


class DefaultPostProcessor(BasePostProcessor):
    """Simple post‑processor that crafts a narrative answer.

    For brevity this converts *raw_result* to string.  It also appends SQL when
    *technical* is True.
    """

    MAX_ROWS_INLINE = 15

    def process(
        self,
        user_query: str,
        sql_query: str,
        raw_result: Any,
        technical: bool = False,
    ) -> str:
        # Attempt to format a small tabular result neatly.
        try:
            from tabulate import tabulate  # optional dependency for nicer tables
        except ImportError:  # pragma: no cover
            tabulate = None  # type: ignore

        # Determine if raw_result is a list/tuple of rows
        formatted_result: str
        if isinstance(raw_result, list):
            # If rows are dicts, use keys as headers
            if raw_result and isinstance(raw_result[0], dict):
                headers = raw_result[0].keys()
                rows = [row.values() for row in raw_result]
                formatted_result = (
                    tabulate(rows, headers=headers, tablefmt="github")
                    if tabulate
                    else str(raw_result)
                )
            else:
                formatted_result = (
                    tabulate(raw_result, tablefmt="github")
                    if tabulate
                    else str(raw_result)
                )
        else:
            formatted_result = str(raw_result)

        narrative = (
            f"Here\u2019s what I found based on your request:\n\n{formatted_result}"
        )

        if technical:
            narrative += f"\n\n```sql\n{sql_query}\n```"
        return narrative


# ---------------------------------------------------------------------------
# Public helper for settings‑based loading
# ---------------------------------------------------------------------------


_DEFAULTS = {
    "INTENT_CLASSIFIER": "db_chat.components.DefaultIntentClassifier",
    "POST_PROCESSOR": "db_chat.components.DefaultPostProcessor",
}


def get_component(component_name: str, dotted_path: str | None = None):
    """Return an instantiated component (intent classifier, post‑processor, ...).

    Parameters
    ----------
    component_name: str
        "INTENT_CLASSIFIER" or "POST_PROCESSOR" (used for defaults & logging).
    dotted_path: str | None
        Optional dotted path from Django settings; if *None* we fall back to
        the default for *component_name*.
    """

    path = dotted_path or _DEFAULTS[component_name]
    try:
        cls = import_from_dotted_path(path)
        return cls()  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover
        logger.exception(
            f"Error loading {component_name} from '{path}'. Falling back to default."
        )
        if path != _DEFAULTS[component_name]:
            # second attempt with default dotted path
            cls = import_from_dotted_path(_DEFAULTS[component_name])
            return cls()  # type: ignore[call-arg]
        raise exc
