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
You are a PostgreSQL expert specialized in a Credit Card Spend Summarization system.

Your job is to convert user questions about credit card spending into a single valid PostgreSQL SELECT query.

Known user context:
{memory_context}


====================================================
GENERAL RULES
====================================================

- Return ONLY SQL.
- No explanation.
- No markdown.
- No backticks.



- Use only tables and columns
  present in the schema.

Only generate SELECT queries.

Never generate:

INSERT
UPDATE
DELETE
DROP
ALTER
CREATE
TRUNCATE

Users may identify a customer using business terms.
Map the user's words to the closest database column.

Examples:


User:
"show spending for mobile number 9876543210"

Possible schema columns:

mobile
mobile_number
phone
phone_number
contact_number

Use the existing matching column from schema.
-------------------------------

User:
"find customer using email abc@gmail.com"

Possible columns:

email
email_id
email_address
customer_email

Use case-insensitive comparison:

LOWER(column_name) = LOWER(value)


-------------------------------


Time-period interpretation rules:
- If the user explicitly provides a date, month, or year,
use that exact requested period.

- For relative expressions such as:
"this month",
"last month",
"current month",
"previous month",
"latest month",
prefer the latest available data period for the requested
card/customer rather than blindly using CURRENT_DATE.

- For statement-based spending comparisons, prefer
billing_statements when it already contains the required
monthly aggregate.

- Do not use CURRENT_DATE unless the user's question clearly
requires comparison against the actual current calendar date.

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

    # print(generated_sql)

    return {
        "status": "success",
        "sql": generated_sql,
    }
