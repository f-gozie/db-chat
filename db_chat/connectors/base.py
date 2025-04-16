import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseConnector(ABC):
    """Abstract base class defining the interface for database connectors.

    This class provides a common structure for managing database connections.
    Subclasses must implement the abstract methods for specific database types.

    Attributes:
        connection_string (Optional[str]): The connection string used to connect.
        conn (Optional[Any]): The active database connection object. Managed by subclasses.
        logger (logging.Logger): Logger instance for the connector.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """Initializes the BaseConnector.

        Args:
            connection_string (Optional[str]): The connection string for the database.
                                               Defaults to None.
        """
        self.connection_string: Optional[str] = connection_string
        self.conn: Optional[Any] = None
        self.logger: logging.Logger = logging.getLogger(__name__)

    @abstractmethod
    def connect(self) -> bool:
        """Establish a connection to the database.

        Subclasses must implement this method to handle the specifics
        of connecting to their target database.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection.

        Subclasses must implement this method to properly close the
        database connection and clean up resources.
        """
        pass

    @abstractmethod
    def get_connection(self) -> Optional[Any]:
        """Return the raw database connection object.

        Subclasses must implement this method to provide access to the
        underlying connection object, potentially re-establishing it if needed.

        Returns:
            Optional[Any]: The active connection object, or None if not connected.
        """
        pass

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format query results into a markdown table string.

        Helper method to present database query results in a human-readable format.

        Args:
            results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                            represents a row and keys are column names.

        Returns:
            str: A string formatted as a markdown table, or "No results found." if empty.
        """
        if not results:
            return "No results found."

        columns = list(results[0].keys())
        if not columns:
            return "Query returned rows with no columns."

        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---" for _ in columns]) + " |"
        rows = []
        for row_dict in results:
            row_values = [
                str(row_dict.get(col, "NULL")) for col in columns
            ]  # Handle potential missing keys
            rows.append("| " + " | ".join(row_values) + " |")

        table = "\n".join([header, separator] + rows)
        table += f"\n\n{len(results)} row(s) returned."
        return table
