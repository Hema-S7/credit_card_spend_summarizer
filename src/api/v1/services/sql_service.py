from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.core.db import get_sql_database
from src.core.llm import get_llm


def generate_sql(
    query: str,
    memory_context: str | None = None,
) -> dict[str, Any]:
    """
    Generate SQL from natural language query.

    This function:
    - reads live DB schema
    - asks LLM for SQL
    - returns SQL only

    It does NOT execute SQL.
    """

    llm = get_llm()

    db = get_sql_database()

    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a PostgreSQL expert.

Given the database schema below,
write a single valid SELECT query
that answers the user's question.

Known user context:
{memory_context}


Rules:

- Return ONLY raw SQL.
- No explanation.
- No markdown.
- No backticks.

- Use only tables and columns
  present in the schema.

- Generate only SELECT statements.

- Do NOT generate:
  INSERT
  UPDATE
  DELETE
  DROP
  ALTER
  CREATE

- Always add LIMIT 50 rows
  unless the question requires aggregation.

Database schema:

{schema}
                """,
            ),
            (
                "human",
                """
Question:

{question}
                """,
            ),
        ]
    )

    chain = sql_prompt | llm

    response = chain.invoke(
        {
            "schema": schema_info,
            "question": query,
            "memory_context": (memory_context or "None"),
        }
    )

    generated_sql = response.content.strip()

    return {
        "status": "success",
        "sql": generated_sql,
    }
