from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.core.db import get_sql_database
from src.core.llm import get_llm


def generate_sql(
    query: str,
    memory_context: str | None = None,
    agent_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate SQL from the analytics/data portion of a natural-language query.

    This function:
    - reads the live DB schema
    - receives the agent's query classification
    - asks the LLM for SQL
    - returns SQL only
    - does NOT execute SQL

    Important:
    For combined queries, the FAQ portion must NOT be converted
    into SQL. The SQL generator handles only the analytics/data
    portion. The FAQ branch handles knowledge-base questions.
    """

    llm = get_llm()

    db = get_sql_database()

    schema_info = db.get_table_info()

    agent_decision = agent_decision or {}

    query_type = agent_decision.get(
        "query_type",
        "analytics",
    )

    needs_faq = agent_decision.get(
        "needs_faq",
        False,
    )

    needs_analytics = agent_decision.get(
        "needs_analytics",
        True,
    )

    retrieval_strategy = agent_decision.get(
        "retrieval_strategy",
        "semantic",
    )

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a PostgreSQL expert specialized in a Credit Card Spend
Summarization system.

Your job is to convert ONLY the ANALYTICS / DATABASE portion of
the user's request into a single valid PostgreSQL SELECT query.

The system may receive combined queries containing both:

1. FAQ / knowledge-base intent
2. Analytics / database intent

IMPORTANT:

If the query is combined, DO NOT attempt to answer the FAQ portion
using SQL.

The FAQ portion is handled separately by the FAQ retrieval branch.

Your responsibility is ONLY the database/analytics portion.

====================================================
QUERY CLASSIFICATION
====================================================

Query type:
{query_type}

Needs FAQ:
{needs_faq}

Needs analytics:
{needs_analytics}

Retrieval strategy:
{retrieval_strategy}

====================================================
GENERAL RULES
====================================================

- Return ONLY SQL.
- No explanation.
- No markdown.
- No backticks.
- Generate exactly one SQL SELECT statement.
- Never generate multiple SQL statements.
- Use only tables and columns present in the schema.
- Only generate SELECT queries.

Never generate:

INSERT
UPDATE
DELETE
DROP
ALTER
CREATE
TRUNCATE

====================================================
COMBINED QUERY RULE
====================================================

For a combined request such as:

"explain cashback benefits and show Mary's credit cards"

the request contains:

FAQ intent:
"explain cashback benefits"

Analytics intent:
"show Mary's credit cards"

The SQL generator MUST generate SQL ONLY for:

"show Mary's credit cards"

It MUST NOT try to retrieve cashback benefits using SQL.

The FAQ branch will independently retrieve the cashback information.

Another example:

User:
"what are the reward benefits and how much did James spend in March?"

SQL responsibility:

"how much did James spend in March?"

Do NOT generate SQL for:

"what are the reward benefits?"

====================================================
NO ANALYTICS INTENT
====================================================

If needs_analytics is false, do not invent an analytics query.

Return:

SELECT 1;

This is only a safety fallback and should normally not be called
when there is no analytics intent.

====================================================
CUSTOMER IDENTIFICATION
====================================================

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

Use the existing matching column from the schema.

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

====================================================
CUSTOMER NAME MATCHING
====================================================

When the user provides a full customer name such as:

"Mary Smith"

prefer an exact case-insensitive match:

LOWER(full_name) = LOWER('Mary Smith')

When the user provides ONLY a first name such as:

"Mary"

do not require the entire full_name to equal "Mary".

A customer may be stored as:

Mary Smith
Mary Johnson
Mary Brown

In that situation, match the first name while still using
case-insensitive comparison.

For PostgreSQL, a suitable pattern is:

(
    LOWER(full_name) = LOWER('Mary')
    OR
    LOWER(split_part(full_name, ' ', 1)) = LOWER('Mary')
)

Do NOT blindly use:

LOWER(full_name) = LOWER('Mary')

when the user supplied only a first name.

Do NOT use broad:

full_name ILIKE '%Mary%'

because that can match Mary's name appearing elsewhere in the
full name.

When retrieving customer information using a name, always include:

- customer_id
- full_name

along with the requested fields.

====================================================
CARD INFORMATION
====================================================

When the user asks to show a customer's credit cards, retrieve
the relevant card fields from the credit_cards table.

At minimum include:

- customer_id
- full_name
- card_id
- card_variant

Include additional useful card fields when available in the schema,
such as:

- credit_limit
- available_limit
- cash_limit
- outstanding_amt
- statement_date
- due_date
- min_due
- reward_points
- status
- issued_date

Do not fabricate columns. Use only columns present in the schema.

====================================================
TIME PERIOD INTERPRETATION
====================================================

- If the user explicitly provides a date, month, or year,
  use that exact requested period.

- For relative expressions such as:

  "this month"
  "last month"
  "current month"
  "previous month"
  "latest month"

  prefer the latest available data period for the requested
  card/customer rather than blindly using CURRENT_DATE.

- For statement-based spending comparisons, prefer billing_statements
  when it already contains the required monthly aggregate.

- Do not use CURRENT_DATE unless the user's question clearly
  requires comparison against the actual current calendar date.

====================================================
SQL SAFETY
====================================================

Always add:

LIMIT 50

unless the question requires aggregation.

For aggregation queries, do not add LIMIT unless appropriate.

====================================================
DATABASE SCHEMA
====================================================

{schema}

====================================================
USER CONTEXT
====================================================

{memory_context}
                """,
            ),
            (
                "human",
                """
Original user question:

{question}

IMPORTANT:

Generate SQL only for the analytics/database intent.

If this is a combined query, ignore the FAQ/knowledge-base
portion because it is handled by another branch.
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
            "query_type": query_type,
            "needs_faq": needs_faq,
            "needs_analytics": needs_analytics,
            "retrieval_strategy": retrieval_strategy,
        }
    )

    generated_sql = response.content.strip()

    return {
        "status": "success",
        "sql": generated_sql,
    }
