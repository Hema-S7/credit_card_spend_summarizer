import re
from typing import Any

FORBIDDEN_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
}


def sanitize_sql(
    sql: str,
) -> str:
    """
    Basic SQL cleanup before validation.
    """

    sql = sql.strip()

    # Remove markdown fences if LLM accidentally adds them
    sql = re.sub(
        r"```sql",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = sql.replace(
        "```",
        "",
    )

    return sql.strip()


def validate_single_statement(
    sql: str,
) -> bool:
    """
    Ensure only one SQL statement exists.
    """

    statements = [s.strip() for s in sql.split(";") if s.strip()]

    return len(statements) == 1


def validate_read_only(
    sql: str,
) -> bool:
    """
    Ensure query is read-only SELECT.
    """
    normalized = sql.upper()

    starts_read_only = normalized.startswith("SELECT") or normalized.startswith("WITH")

    if not starts_read_only:
        return False

    for keyword in FORBIDDEN_KEYWORDS:

        if re.search(
            rf"\b{keyword}\b",
            normalized,
        ):
            return False

    return True


def extract_tables(
    sql: str,
) -> list[str]:
    """
    Extract table names from SQL.

    Basic extraction for validation/audit.
    """

    tables = []

    matches = re.findall(
        r"\bFROM\s+([a-zA-Z0-9_\.]+)" r"|\bJOIN\s+([a-zA-Z0-9_\.]+)",
        sql,
        flags=re.IGNORECASE,
    )

    for match in matches:

        table = match[0] or match[1]

        if table:
            tables.append(table)

    return list(set(tables))


def validate_sql(
    sql: str,
    schema_tables: list[str] | None = None,
) -> dict[str, Any]:
    """
    Complete SQL validation pipeline.
    """

    issues = []

    cleaned_sql = sanitize_sql(sql)

    single_statement = validate_single_statement(cleaned_sql)

    read_only = validate_read_only(cleaned_sql)

    tables_used = extract_tables(cleaned_sql)

    schema_valid = True

    if schema_tables:

        for table in tables_used:

            if table not in schema_tables:

                schema_valid = False

                issues.append(f"Unknown table: {table}")

    if not single_statement:

        issues.append("Multiple SQL statements are not allowed.")

    if not read_only:

        issues.append("Only SELECT statements are allowed.")

    passed = single_statement and read_only and schema_valid

    return {
        "passed": passed,
        "is_read_only": read_only,
        "syntax_valid": True,
        "schema_valid": schema_valid,
        "single_statement": single_statement,
        "tables_used": tables_used,
        "columns_used": [],
        "issues": issues,
    }
