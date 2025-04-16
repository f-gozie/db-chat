import logging

import psycopg2
import psycopg2.extras

from ..base import BaseConnector

logger = logging.getLogger(__name__)


class PgConnector(BaseConnector):
    """Manages the connection to a PostgreSQL database.

    This class handles the lifecycle of a PostgreSQL database connection,
    including establishing, closing, and providing access to the connection object.
    It inherits from `BaseConnector`.

    Attributes:
        connection_string (str): The database connection string.
        conn (psycopg2.connection | None): The active database connection object,
                                          or None if not connected.
    """

    def __init__(self, connection_string: str):
        """Initializes the PgConnector.

        Args:
            connection_string (str): The connection string for the PostgreSQL database.
                                    Must be a valid PostgreSQL connection string.
        """
        super().__init__(connection_string)

    def connect(self) -> bool:
        """Establishes a connection to the PostgreSQL database.

        Uses the `connection_string` provided during initialization.
        If a connection already exists and is open, it logs and returns True.
        Otherwise, attempts to establish a new connection.

        Returns:
            bool: True if the connection is successfully established or already exists,
                  False if connection fails.
        """
        if self.conn and not self.conn.closed:
            logger.info("Connection already established.")
            return True
        try:
            logger.info(
                f"Connecting to PostgreSQL using the provided connection string."
            )  # Avoid logging the full string
            self.conn = psycopg2.connect(self.connection_string)
            logger.info("Successfully connected to PostgreSQL.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            self.conn = None  # Ensure conn is None on failure
            return False

    def disconnect(self):
        """Closes the active database connection.

        If a connection exists, it attempts to close it and sets `self.conn` to None.
        Logs information about the disconnection attempt or if no connection exists.
        """
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                logger.info("Disconnected from PostgreSQL.")
            except Exception as e:
                logger.error(f"Error disconnecting from PostgreSQL: {str(e)}")
        else:
            logger.info("No active connection to disconnect.")

    def get_connection(self):
        """Retrieves the active database connection object.

        Checks if the current connection is active. If not, it attempts to
        re-establish the connection using `connect()`.

        Returns:
            psycopg2.connection | None: The active database connection object,
                                        or None if the connection is closed and
                                        cannot be re-established.
        """
        if not self.conn or self.conn.closed:
            logger.warning(
                "Connection is closed or not established. Attempting to reconnect."
            )
            if not self.connect():
                return None  # Failed to reconnect
        return self.conn
