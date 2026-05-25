from __future__ import annotations

import os
import json
import sqlite3

from typing import Optional, List, Any, Dict
from pathlib import Path

from pydantic import BaseModel, Field


# =========================================================
# CONFIGURATION
# Read from environment variables. Set these in your .env:
#
#   DB_ENGINE      sqlite | postgres | mysql  (default: sqlite)
#   DB_PATH        path to SQLite file        (sqlite only)
#   DB_HOST        database host
#   DB_PORT        database port
#   DB_NAME        database name
#   DB_USER        database user
#   DB_PASSWORD    database password
# =========================================================

def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# =========================================================
# SCHEMAS
# =========================================================

class ExecuteSQLSchema(BaseModel):
    query: str = Field(..., description="SQL query to execute.")
    params: Optional[List[Any]] = Field(None, description="Optional positional parameters for parameterized queries.")
    max_rows: int = Field(100, ge=1, le=5000, description="Maximum rows to return for SELECT queries.")


class ListTablesSchema(BaseModel):
    schema_name: Optional[str] = Field(None, description="Schema/database name to list tables from. Uses default if not set.")


class DescribeTableSchema(BaseModel):
    table_name: str = Field(..., description="Name of the table to describe.")
    schema_name: Optional[str] = Field(None, description="Schema/database name. Uses default if not set.")


# =========================================================
# CONNECTION FACTORY
# =========================================================

def _get_connection():
    """
    Return a DB-API 2.0 connection based on DB_ENGINE env var.
    Supports SQLite, PostgreSQL (psycopg2), and MySQL (mysql-connector-python).
    """

    engine = _cfg("DB_ENGINE", "sqlite").lower()

    if engine == "sqlite":
        db_path = _cfg("DB_PATH", "agent.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn, "sqlite"

    elif engine == "postgres":
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL. "
                "Install with: pip install psycopg2-binary"
            )
        conn = psycopg2.connect(
            host=_cfg("DB_HOST", "localhost"),
            port=int(_cfg("DB_PORT", "5432")),
            dbname=_cfg("DB_NAME", "postgres"),
            user=_cfg("DB_USER", "postgres"),
            password=_cfg("DB_PASSWORD", ""),
        )
        return conn, "postgres"

    elif engine == "mysql":
        try:
            import mysql.connector
        except ImportError:
            raise ImportError(
                "mysql-connector-python is required for MySQL. "
                "Install with: pip install mysql-connector-python"
            )
        conn = mysql.connector.connect(
            host=_cfg("DB_HOST", "localhost"),
            port=int(_cfg("DB_PORT", "3306")),
            database=_cfg("DB_NAME", ""),
            user=_cfg("DB_USER", "root"),
            password=_cfg("DB_PASSWORD", ""),
        )
        return conn, "mysql"

    else:
        raise ValueError(
            f"Unsupported DB_ENGINE: '{engine}'. "
            "Use 'sqlite', 'postgres', or 'mysql'."
        )


def _rows_to_table(columns: List[str], rows: List[tuple]) -> str:
    """
    Format query results as a plain-text table.
    """

    if not rows:
        return "(no rows)"

    col_widths = [
        max(len(str(col)), max((len(str(r[i])) for r in rows), default=0))
        for i, col in enumerate(columns)
    ]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header = (
        "|"
        + "|".join(f" {col:<{col_widths[i]}} " for i, col in enumerate(columns))
        + "|"
    )

    data_rows = []
    for row in rows:
        data_rows.append(
            "|"
            + "|".join(
                f" {str(row[i]):<{col_widths[i]}} "
                for i in range(len(columns))
            )
            + "|"
        )

    return "\n".join([sep, header, sep] + data_rows + [sep])


# =========================================================
# IMPLEMENTATIONS
# =========================================================

def execute_sql(
    query: str,
    params: Optional[List[Any]] = None,
    max_rows: int = 100,
) -> str:
    """
    Execute a SQL query and return the results as a formatted table.
    Supports SELECT (returns rows) and DML/DDL (returns rows affected).
    Uses parameterized queries to prevent SQL injection.
    """

    query = query.strip()

    if not query:
        return "ERROR: Empty query."

    # Warn on dangerous patterns without WHERE
    upper = query.upper()
    is_destructive = any(upper.startswith(kw) for kw in ("DELETE", "UPDATE", "DROP", "TRUNCATE"))
    if is_destructive and "WHERE" not in upper:
        return (
            "SAFETY BLOCK: Destructive query with no WHERE clause detected.\n"
            f"Query: {query}\n"
            "Add a WHERE clause to proceed, or confirm this is intentional."
        )

    try:
        conn, engine = _get_connection()
    except Exception as e:
        return f"ERROR: Could not connect to database: {e}"

    try:
        cursor = conn.cursor()
        cursor.execute(query, params or [])

        is_select = upper.startswith(("SELECT", "WITH", "SHOW", "EXPLAIN", "PRAGMA"))

        if is_select:
            columns = [desc[0] for desc in cursor.description or []]
            all_rows = cursor.fetchmany(max_rows + 1)
            truncated = len(all_rows) > max_rows
            rows = all_rows[:max_rows]

            # Convert Row objects to plain tuples
            plain_rows = [tuple(r) for r in rows]

            table = _rows_to_table(columns, plain_rows)

            result = (
                f"Query OK — {len(rows)} row(s) returned"
                + (f" (limited to {max_rows}, more rows exist)" if truncated else "")
                + f"\n\n{table}"
            )

        else:
            conn.commit()
            affected = cursor.rowcount
            result = (
                f"Query OK — {affected} row(s) affected."
            )

        return result

    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return f"SQL ERROR: {e}\nQuery: {query}"

    finally:
        try:
            conn.close()
        except Exception:
            pass


def list_tables(
    schema_name: Optional[str] = None,
) -> str:
    """
    List all tables in the database (or a specific schema).
    Returns table names formatted as a list.
    """

    try:
        conn, engine = _get_connection()
    except Exception as e:
        return f"ERROR: Could not connect to database: {e}"

    try:
        cursor = conn.cursor()

        if engine == "sqlite":
            cursor.execute(
                "SELECT name, type FROM sqlite_master "
                "WHERE type IN ('table', 'view') ORDER BY name;"
            )
            rows = cursor.fetchall()
            if not rows:
                return "No tables found."
            lines = [f"  {row[1].upper():<6}  {row[0]}" for row in rows]
            return (
                f"Tables in SQLite DB ({_cfg('DB_PATH', 'agent.db')}):\n"
                + "\n".join(lines)
            )

        elif engine == "postgres":
            schema = schema_name or "public"
            cursor.execute(
                "SELECT table_name, table_type "
                "FROM information_schema.tables "
                "WHERE table_schema = %s ORDER BY table_name;",
                [schema],
            )
            rows = cursor.fetchall()
            if not rows:
                return f"No tables found in schema '{schema}'."
            lines = [f"  {row[1]:<20}  {row[0]}" for row in rows]
            return f"Tables in schema '{schema}':\n" + "\n".join(lines)

        elif engine == "mysql":
            db = schema_name or _cfg("DB_NAME")
            cursor.execute(f"SHOW FULL TABLES FROM `{db}`;")
            rows = cursor.fetchall()
            if not rows:
                return f"No tables found in database '{db}'."
            lines = [f"  {row[1]:<20}  {row[0]}" for row in rows]
            return f"Tables in database '{db}':\n" + "\n".join(lines)

        return "ERROR: Unknown engine."

    except Exception as e:
        return f"ERROR listing tables: {e}"

    finally:
        try:
            conn.close()
        except Exception:
            pass


def describe_table(
    table_name: str,
    schema_name: Optional[str] = None,
) -> str:
    """
    Return the column names, types, nullability, and key info for a table.
    """

    try:
        conn, engine = _get_connection()
    except Exception as e:
        return f"ERROR: Could not connect to database: {e}"

    try:
        cursor = conn.cursor()

        if engine == "sqlite":
            cursor.execute(f"PRAGMA table_info(`{table_name}`);")
            rows = cursor.fetchall()
            if not rows:
                return f"ERROR: Table '{table_name}' not found."
            lines = ["cid  name                  type          notnull  dflt_value  pk"]
            lines.append("-" * 65)
            for row in rows:
                lines.append(
                    f"{row[0]:<4} {row[1]:<22} {row[2]:<13} "
                    f"{row[3]:<8} {str(row[4]):<11} {row[5]}"
                )
            return f"Schema for '{table_name}':\n" + "\n".join(lines)

        elif engine == "postgres":
            schema = schema_name or "public"
            cursor.execute(
                "SELECT column_name, data_type, is_nullable, "
                "column_default, character_maximum_length "
                "FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = %s "
                "ORDER BY ordinal_position;",
                [schema, table_name],
            )
            rows = cursor.fetchall()
            if not rows:
                return f"ERROR: Table '{table_name}' not found in schema '{schema}'."
            lines = ["column_name            data_type             nullable  default"]
            lines.append("-" * 75)
            for row in rows:
                lines.append(
                    f"{row[0]:<23} {row[1]:<21} {row[2]:<9} {str(row[3] or '')}"
                )
            return f"Schema for '{schema}.{table_name}':\n" + "\n".join(lines)

        elif engine == "mysql":
            cursor.execute(f"DESCRIBE `{table_name}`;")
            rows = cursor.fetchall()
            if not rows:
                return f"ERROR: Table '{table_name}' not found."
            lines = ["Field                 Type                  Null  Key  Default  Extra"]
            lines.append("-" * 75)
            for row in rows:
                lines.append(
                    f"{str(row[0]):<22} {str(row[1]):<22} {str(row[2]):<5} "
                    f"{str(row[3]):<4} {str(row[4]):<8} {str(row[5])}"
                )
            return f"Schema for '{table_name}':\n" + "\n".join(lines)

        return "ERROR: Unknown engine."

    except Exception as e:
        return f"ERROR describing table: {e}"

    finally:
        try:
            conn.close()
        except Exception:
            pass


# =========================================================
# TOOL REGISTRY ENTRIES
# =========================================================

SQL_TOOLS = {
    "execute_sql": {
        "func": execute_sql,
        "schema": ExecuteSQLSchema,
        "description": (
            "Execute a SQL query (SELECT, INSERT, UPDATE, DELETE, DDL). "
            "Supports parameterized queries. Blocks destructive queries "
            "without a WHERE clause. Configured via DB_ENGINE, DB_PATH, "
            "DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD env vars."
        ),
    },
    "list_tables": {
        "func": list_tables,
        "schema": ListTablesSchema,
        "description": (
            "List all tables and views in the connected database. "
            "Supports SQLite, PostgreSQL, and MySQL."
        ),
    },
    "describe_table": {
        "func": describe_table,
        "schema": DescribeTableSchema,
        "description": (
            "Return the column definitions (name, type, nullability, "
            "defaults, keys) for a specific table."
        ),
    },
}
