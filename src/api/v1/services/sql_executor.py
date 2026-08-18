from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.core.config import (
    MAX_RESULT_ROWS,
    QUERY_TIMEOUT_SECONDS,
)

from src.core.database import (
    get_raw_connection,
)


def execute_sql(
    sql: str,
) -> dict[str, Any]:
    """
    Execute validated SQL safely.

    Returns SQLExecutionResult structure.
    """

    try:

        with get_raw_connection() as conn:

            # PostgreSQL statement timeout
            with conn.cursor(row_factory=dict_row) as cur:

                cur.execute(f"""
                    SET LOCAL statement_timeout = {QUERY_TIMEOUT_SECONDS * 1000};
                    """)

                cur.execute(sql)

                rows = cur.fetchall()

                columns = list(rows[0].keys()) if rows else []

                truncated = len(rows) > MAX_RESULT_ROWS

                limited_rows = rows[:MAX_RESULT_ROWS]

                if not limited_rows:

                    return {
                        "status": "empty",
                        "columns": columns,
                        "rows": [],
                        "row_count": 0,
                        "truncated": False,
                        "error_code": None,
                        "error_message": None,
                    }

                return {
                    "status": "success",
                    "columns": columns,
                    "rows": limited_rows,
                    "row_count": len(limited_rows),
                    "truncated": truncated,
                    "error_code": None,
                    "error_message": None,
                }

    except psycopg.errors.QueryCanceled:

        return {
            "status": "timeout",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error_code": "QUERY_TIMEOUT",
            "error_message": (
                "The database query exceeded " "the allowed execution time."
            ),
        }

    except Exception:

        return {
            "status": "error",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error_code": ("QUERY_EXECUTION_ERROR"),
            "error_message": ("The database query " "could not be executed."),
        }
