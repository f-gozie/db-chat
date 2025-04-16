"""SQL queries for PostgreSQL schema introspection."""

GET_COLUMNS_QUERY = """
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM
    information_schema.columns
WHERE
    table_name = %s
ORDER BY
    ordinal_position;
"""

GET_PRIMARY_KEYS_QUERY = """
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
    tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = %s;
"""

GET_FOREIGN_KEYS_QUERY = """
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
    tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
"""

GET_CHECK_CONSTRAINTS_QUERY = """
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
    AND pgc.conrelid::regclass::text = %s;
"""
