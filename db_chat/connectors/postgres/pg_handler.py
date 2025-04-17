import logging
import re

import psycopg2
import psycopg2.extras

from db_chat.model_registry import get_registry

from ..handlers.base import BaseHandler
from . import queries

logger = logging.getLogger(__name__)


class PgHandler(BaseHandler):
    """PostgreSQL handler implementation.

    Uses a PgConnector to interact with the database.
    """

    def execute_query(self, sql_query, params=None):
        """Execute a read-only SQL query and return formatted results.

        This method executes a SQL query against the PostgreSQL database and returns
        the results in a formatted string (typically as a markdown table).

        Args:
            sql_query (str): The SQL query to execute.
            params (tuple, optional): Parameters to pass to the query for safe parameterization.
                                     Defaults to None.

        Returns:
            str: A formatted string containing the query results as a markdown table,
                 or an error message if the query execution fails.

        Note:
            This method enforces read-only transactions for safety and will not
            commit any data modifications to the database.
        """
        conn = self.connector.get_connection()
        if not conn:
            return "Error: Failed to get database connection"

        final_sql = sql_query
        final_params = params or ()

        # Escape literal % signs if no parameters are provided, to avoid psycopg2 misinterpretation
        if not final_params:
            final_sql = sql_query.replace("%", "%%")

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("BEGIN TRANSACTION READ ONLY")
                cursor.execute(final_sql, final_params)

                if cursor.description:
                    try:
                        result_list = cursor.fetchall()
                        formatted_result = self._format_results(result_list)
                        logger.debug(f"Formatted response: {formatted_result}")
                        cursor.execute("COMMIT")
                        return formatted_result
                    except Exception as fetch_err:
                        logger.error(
                            f"Error during fetchall() or formatting: {fetch_err}, cursor.description: {getattr(cursor, 'description', None)}"
                        )
                        return (
                            f"Error: Failed to process query results - {str(fetch_err)}"
                        )
                else:
                    return "Query executed successfully, but it did not return any results."

        except Exception as e:
            logger.error(
                f"Error executing SQL query: {e}, Original SQL: {sql_query}, Params: {params}"
            )
            return f"Error: SQL execution failed - {str(e)}"

    def get_schema(self, tables=None):
        """Fetch and format schema information for specified tables.

        This method retrieves detailed schema information for the requested tables,
        including column names, data types, constraints, and relationships.
        It first attempts to get schema information from the Django model registry,
        and falls back to direct database inspection for tables not found in the registry.

        Args:
            tables (list, optional): A list of table names to retrieve schema for.
                                    If None, an error message is returned.

        Returns:
            str: A formatted string containing the schema information for the requested tables.
                 Returns an error message if tables is None or if an error occurs during retrieval.

        Raises:
            No exceptions are raised directly; errors are caught and returned as strings.
        """
        if not tables:
            return "Error: No tables specified for schema retrieval."

        try:
            registry = get_registry()
            model_schema_info = []
            db_schema_tables = []

            for table in tables:
                model_schema = registry.get_table_schema(table)
                if model_schema:
                    model_schema_info.append(model_schema)
                else:
                    db_schema_tables.append(table)

            db_schema_info_str = ""
            if db_schema_tables:
                db_schema_info = self._get_db_schema(db_schema_tables)
                if db_schema_info and not db_schema_info.startswith("Error:"):
                    db_schema_info_str = db_schema_info
                elif db_schema_info:
                    return db_schema_info

            all_schema_parts = model_schema_info
            if db_schema_info_str:
                all_schema_parts.append(db_schema_info_str)

            if not all_schema_parts:
                return "No schema information available for the specified tables."

            return "\n\n".join(all_schema_parts)

        except ImportError:
            logger.info(
                "Model registry not available, falling back to DB schema retrieval."
            )
            return self._get_db_schema(tables)
        except Exception as e:
            logger.error(f"Error fetching schema: {e}")
            return f"Error: {str(e)}"

    def _get_db_schema(self, tables):
        """Retrieve and format schema information directly from the database.

        Connects to the database and executes SQL queries defined in `queries.py`
        to fetch column details, primary keys, foreign keys, and check constraints
        for the specified list of tables.

        Args:
            tables (list[str]): A list of table names for which to retrieve schema.

        Returns:
            str: A formatted string containing the combined schema information for
                 all requested tables, or an error message if connection or query fails,
                 or if no schema info is found for any specified tables.
        """
        conn = self.connector.get_connection()
        if not conn:
            return "Error: Failed to get database connection for schema retrieval"

        schema_info = []
        try:
            with conn.cursor() as cursor:
                for table in tables:
                    cursor.execute(queries.GET_COLUMNS_QUERY, (table,))
                    columns = cursor.fetchall()
                    if not columns:
                        logger.warning(
                            f"No columns found for table '{table}'. Skipping."
                        )
                        continue

                    cursor.execute(queries.GET_PRIMARY_KEYS_QUERY, (table,))
                    pks = [pk[0] for pk in cursor.fetchall()]

                    cursor.execute(queries.GET_FOREIGN_KEYS_QUERY, (table,))
                    fks = cursor.fetchall()
                    fk_dict = {fk[0]: (fk[1], fk[2]) for fk in fks}

                    cursor.execute(queries.GET_CHECK_CONSTRAINTS_QUERY, (table,))
                    check_constraints = cursor.fetchall()
                    choice_fields = self._extract_choices_from_constraints(
                        check_constraints
                    )

                    table_schema_str = self._format_table_schema(
                        table, pks, columns, fk_dict, choice_fields
                    )
                    schema_info.append(table_schema_str)
        except Exception as e:
            logger.error(f"Error fetching schema directly from database: {e}")
            return f"Error fetching schema from database: {str(e)}"

        if not schema_info:
            return "No schema information could be retrieved from the database for the specified tables."

        return "\n\n".join(schema_info)

    def _extract_choices_from_constraints(self, check_constraints):
        """Parse choices from PostgreSQL CHECK constraint definitions.

        This helper function uses regular expressions to identify and extract
        field names and their allowed values from CHECK constraint definitions,
        specifically targeting the pattern commonly used by Django's choices fields.

        Args:
            check_constraints (list[tuple]): A list of tuples, where each tuple
                                            represents a check constraint. The third
                                            element (index 2) is expected to be the
                                            constraint definition string.

        Returns:
            dict[str, list[str]]: A dictionary where keys are field names and values
                                  are lists of allowed choices for that field.
        """
        choice_fields = {}
        for constraint in check_constraints:
            # constraint_definition is typically at index 2
            constraint_def = constraint[2]
            # Regex for Django's common CHECK constraint pattern for choices
            # Example: ((("status")::text = ANY (ARRAY[('PENDING'::text), ('COMPLETED'::text)])))
            match = re.search(
                r'\(\("?([a-zA-Z0-9_]+)"?\)::text = ANY \(ARRAY\[(.*?)\]\)\)',
                constraint_def,
            )
            if match:
                field_name = match.group(1)
                values_str = match.group(2)
                # Extract values within single quotes
                values = re.findall(r"'([^']*)'", values_str)
                if values:
                    choice_fields[field_name] = values
        return choice_fields

    def _format_table_schema(self, table_name, pks, columns, fk_dict, choice_fields):
        """Format the schema details of a single table into a readable string.

        Constructs a multi-line string representation of a table's schema,
        including its name, primary keys, and details for each column (name,
        data type, nullability, default value, foreign key references, and choices).

        Args:
            table_name (str): The name of the table.
            pks (list[str]): A list of primary key column names.
            columns (list[tuple]): A list of tuples describing each column, typically
                                   containing (column_name, data_type, is_nullable, default_value).
            fk_dict (dict[str, tuple[str, str]]): A dictionary mapping column names to tuples
                                                  containing (foreign_table_name, foreign_column_name).
            choice_fields (dict[str, list[str]]): A dictionary mapping column names to lists
                                                  of allowed choices derived from check constraints.

        Returns:
            str: A formatted string describing the table's schema.
        """
        table_schema = [f"Table: {table_name} (Primary Keys: {', '.join(pks)})\n"]
        for col_data in columns:
            col_name, data_type, is_nullable, default = col_data
            nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
            default_str = f" DEFAULT {default}" if default else ""

            fk_info = ""
            if col_name in fk_dict:
                ft_name, ft_col = fk_dict[col_name]
                fk_info = f" REFERENCES {ft_name}({ft_col})"

            choices_info = ""
            if col_name in choice_fields:
                choices_info = f" CHOICES: {', '.join(choice_fields[col_name])}"

            col_def = f"  - {col_name}: {data_type} {nullable}{default_str}{fk_info}{choices_info}"
            table_schema.append(col_def)

        return "\n".join(table_schema)
