"""Direct PostgreSQL executor for the database chat application."""

import logging
import re

import psycopg2
import psycopg2.extras


class PostgreSQLExecutor:
    """Class to handle direct PostgreSQL interactions."""

    def __init__(self, connection_string=None):
        """Initialize the executor with optional connection string."""
        # Default connection string - this should be updated in settings.py
        self.connection_string = (
            connection_string
            or "postgresql://debug:debug@postgres:5432/eyemark_backend"
        )
        self.conn = None
        self.logger = logging.getLogger(__name__)

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            self.logger.info(f"Connecting to PostgreSQL with: {self.connection_string}")
            self.conn = psycopg2.connect(self.connection_string)
            self.logger.info("Successfully connected to PostgreSQL")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            return False

    def disconnect(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.logger.info("Disconnected from PostgreSQL")

    def execute_query(self, sql_query):
        """Execute a read-only SQL query and return the results.

        Args:
            sql_query (str): The SQL query to execute

        Returns:
            dict: A dictionary with 'result' containing formatted results or an error message
        """
        if not self.conn:
            if not self.connect():
                return {"result": "Error: Failed to connect to database"}

        # Ensure the connection is still alive
        try:
            if self.conn.closed:
                self.connect()
        except:
            self.connect()

        try:
            # Set the transaction to read-only for safety
            cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Execute the query in a read-only transaction
            cursor.execute("BEGIN TRANSACTION READ ONLY")
            cursor.execute(sql_query)

            # Fetch results (if any)
            if cursor.description:
                results = cursor.fetchall()
                # Convert to list of dicts for easier JSON serialization
                result_list = [dict(row) for row in results]

                # Format results nicely
                formatted_result = self._format_results(result_list)
                cursor.execute("COMMIT")
                return {"result": formatted_result}
            else:
                # No results (like for CREATE/INSERT/etc which shouldn't happen in read-only mode)
                cursor.execute("COMMIT")
                return {"result": "Query executed successfully (no results)"}

        except Exception as e:
            # Rollback transaction on error
            if self.conn and not self.conn.closed:
                self.conn.rollback()

            error_message = f"Error executing SQL query: {str(e)}"
            self.logger.error(error_message)
            return {"result": error_message}

    def _format_results(self, results):
        """Format the query results into a readable string.

        Args:
            results (list): List of dictionaries representing rows

        Returns:
            str: Formatted string representation of results
        """
        if not results:
            return "No results found."

        # Calculate column widths for pretty formatting
        columns = results[0].keys()

        # Create a markdown table
        table = "| " + " | ".join(columns) + " |\n"
        table += "| " + " | ".join(["---" for _ in columns]) + " |\n"

        # Add data rows
        for row in results:
            table += "| " + " | ".join([str(row[col]) for col in columns]) + " |\n"

        # Add result count
        table += f"\n{len(results)} rows returned."

        return table

    def get_schema(self, tables=None):
        """Fetches the schema for specified tables.

        Args:
            tables: List of tables to fetch schema for. If None, fetches all tables.

        Returns:
            str: Formatted schema information
        """
        try:
            # First, try to get schema from Django model registry which has accurate choices
            try:
                from .model_registry import get_registry

                registry = get_registry()

                model_schema_info = []
                db_schema_info = []
                missing_tables = []

                # Try to get schema from models first
                for table in tables:
                    model_schema = registry.get_table_schema(table)
                    if model_schema:
                        model_schema_info.append(model_schema)
                    else:
                        missing_tables.append(table)

                # For tables not found in models, use database inspection
                if missing_tables:
                    db_schema = self._get_db_schema(missing_tables)
                    if db_schema and not db_schema.startswith("Error:"):
                        db_schema_info.append(db_schema)

                # Combine model and DB schema information
                all_schema_info = model_schema_info + db_schema_info

                if not all_schema_info:
                    return "No schema information available for the specified tables."

                return "\n\n".join(all_schema_info)

            except ImportError:
                # Fall back to direct database inspection if model registry isn't available
                return self._get_db_schema(tables)

        except Exception as e:
            logger.error(f"Error fetching schema: {e}")
            return f"Error: {str(e)}"

    def _get_db_schema(self, tables=None):
        """Gets schema directly from database using SQL queries."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            schema_info = []

            for table in tables:
                # Get basic column information
                cursor.execute(
                    f"""
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM
                        information_schema.columns
                    WHERE
                        table_name = '{table}'
                    ORDER BY
                        ordinal_position;
                """
                )

                columns = cursor.fetchall()

                if not columns:
                    continue

                # Get primary key information
                cursor.execute(
                    f"""
                    SELECT
                        c.column_name
                    FROM
                        information_schema.table_constraints tc
                    JOIN
                        information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                    JOIN
                        information_schema.columns AS c ON c.table_schema = tc.constraint_schema
                        AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
                    WHERE
                        tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = '{table}';
                """
                )

                pks = [pk[0] for pk in cursor.fetchall()]

                # Get foreign key information
                cursor.execute(
                    f"""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM
                        information_schema.table_constraints AS tc
                    JOIN
                        information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name
                        AND tc.constraint_schema = kcu.constraint_schema
                    JOIN
                        information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name
                        AND ccu.constraint_schema = tc.constraint_schema
                    WHERE
                        tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = '{table}';
                """
                )

                fks = cursor.fetchall()
                fk_dict = {fk[0]: (fk[1], fk[2]) for fk in fks}

                # Get check constraints (often used for choices/enums)
                cursor.execute(
                    f"""
                    SELECT
                        pgc.conname AS constraint_name,
                        pgc.conrelid::regclass AS table_name,
                        pg_get_constraintdef(pgc.oid) AS constraint_definition
                    FROM
                        pg_constraint pgc
                    JOIN
                        pg_namespace pgn ON pgn.oid = pgc.connamespace
                    WHERE
                        pgc.contype = 'c'
                        AND pgc.conrelid::regclass::text = '{table}'
                """
                )

                check_constraints = cursor.fetchall()

                # Process check constraints to extract choices
                choice_fields = {}
                for constraint in check_constraints:
                    constraint_def = constraint[2]
                    # For common Django pattern: ((("status")::text = ANY (ARRAY[('PENDING'::text, 'COMPLETED'::text, ...)]))
                    match = re.search(
                        r'\(\("?([a-zA-Z0-9_]+)"?\)::text = ANY \(ARRAY\[(.*?)\]\)\)',
                        constraint_def,
                    )
                    if match:
                        field_name = match.group(1)
                        values_str = match.group(2)
                        # Extract the quoted values
                        values = re.findall(r"'([^']*)'", values_str)
                        if values:
                            choice_fields[field_name] = values

                # Add table schema to result
                table_schema = [f"Table: {table} (Primary Keys: {', '.join(pks)})\n"]

                for col in columns:
                    col_name, data_type, is_nullable, default = col
                    nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                    default_str = f" DEFAULT {default}" if default else ""

                    # Add foreign key info if applicable
                    fk_info = ""
                    if col_name in fk_dict:
                        ft_name, ft_col = fk_dict[col_name]
                        fk_info = f" REFERENCES {ft_name}({ft_col})"

                    # Add choices info if applicable
                    choices_info = ""
                    if col_name in choice_fields:
                        choices_info = f" CHOICES: {', '.join(choice_fields[col_name])}"

                    # Build the column definition
                    col_def = f"  - {col_name}: {data_type} {nullable}{default_str}{fk_info}{choices_info}"
                    table_schema.append(col_def)

                schema_info.append("\n".join(table_schema))

            conn.close()

            if not schema_info:
                return "No schema information available for the specified tables."

            return "\n\n".join(schema_info)

        except Exception as e:
            logger.error(f"Error fetching schema from database: {e}")
            return f"Error: {str(e)}"

    def _get_connection(self):
        """Get a PostgreSQL connection object.

        Returns:
            psycopg2.connection: A connection object
        """
        try:
            conn = psycopg2.connect(self.connection_string)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise e
