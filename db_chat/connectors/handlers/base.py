import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..base import BaseConnector  # Assuming BaseConnector is in ../base.py


class BaseHandler(ABC):
    """Abstract base class for database operation handlers.

    Defines a consistent interface for executing database operations (like queries)
    and retrieving database metadata (like schema information), abstracting away the
    specifics of the underlying database type.

    Subclasses are responsible for implementing the abstract methods using a
    concrete `BaseConnector` instance provided during initialization.

    Attributes:
        connector (BaseConnector): An instance of a BaseConnector subclass
            that provides the actual connection to the database.
        logger (logging.Logger): Logger instance specific to the handler subclass.
    """

    def __init__(self, connector: BaseConnector):
        """Initialize the handler with a database connector.

        Args:
            connector (BaseConnector): The database connector instance that provides
                access to the database. Must adhere to the BaseConnector interface.
        """
        self.connector: BaseConnector = connector
        # Use the subclass name for more specific logging
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def execute_query(self, sql_query: str, params: Optional[tuple] = None) -> str:
        """Execute a read-only SQL query and return formatted results.

        Implementations should use the `self.connector` to execute the query
        safely (using parameters if provided) and then format the results
        into a human-readable string (e.g., a markdown table) using
        `_format_results` or a custom formatter.

        Args:
            sql_query (str): The SQL query string to execute.
            params (Optional[tuple]): A tuple of parameters to safely bind to the
                query, preventing SQL injection. Defaults to None if no parameters.

        Returns:
            str: A formatted string containing the query results (e.g., markdown table)
                 or an error message string (should ideally start with "Error:").
        """
        pass

    @abstractmethod
    def get_schema(self, tables: Optional[List[str]] = None) -> str:
        """Fetch and format schema information for specified tables.

        Implementations should use the `self.connector` or specific database
        metadata queries to retrieve information about columns, types, keys,
        and relationships for the requested tables.

        Args:
            tables (Optional[List[str]]): A list of table names for which to retrieve
                schema information. If None, the behavior is implementation-specific
                (e.g., might return schema for all accessible tables or raise an error).

        Returns:
            str: A formatted string describing the schema of the requested tables,
                 or an error message string (should ideally start with "Error:").
        """
        pass

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format query results (list of dictionaries) into a markdown table string.

        A default helper method to convert a list of dictionary-like rows
        (as commonly returned by database cursors like `RealDictCursor`)
        into a simple markdown table suitable for display.

        Args:
            results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                represents a row, and keys represent column names.

        Returns:
            str: A string formatted as a markdown table.
                 Returns "No results found." if the input list is empty.
                 Returns the raw string representation if the input is not a list of dicts.
                 Returns "Query returned rows with no columns." if dicts lack keys.
        """
        if not results:
            return "No results found."

        # Validate input format
        if not isinstance(results, list) or not all(
            isinstance(row, dict) for row in results
        ):
            self.logger.warning(
                "Attempted to format non-dict results. Returning raw data."
            )
            return str(results)

        # Extract column names from the first row (assuming consistency)
        columns = list(results[0].keys())
        if not columns:
            self.logger.warning("Attempted to format results with no columns.")
            return "Query returned rows with no columns."

        # Build markdown table parts
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---" for _ in columns]) + " |"
        rows_str = []
        for row in results:
            # Ensure all columns are present (use .get) and stringified
            row_values = [str(row.get(col, "")) for col in columns]
            rows_str.append("| " + " | ".join(row_values) + " |")

        # Combine parts and add row count
        table = "\n".join([header, separator] + rows_str)
        table += f"\n\n{len(results)} row(s) returned."
        return table
