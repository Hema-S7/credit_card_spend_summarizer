from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.core.config import MAX_RETRIES
from src.core.llm import get_llm


def retry_router(
    state: dict[str, Any],
):
    """
    Decide whether another retry is allowed.
    """

    retry = state.get(
        "retry",
        {
            "count": 0,
            "max_retries": MAX_RETRIES,
        },
    )

    if retry["count"] >= retry["max_retries"]:

        return "exhausted"

    return "retry"


# def retry_rephrase_node(
#     state: dict[str, Any],
# ):

#     llm = get_llm()

#     feedback = state.get(
#         "retry_feedback",
#         {},
#     )

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """
# You are a query refinement assistant.

# Your job is to rewrite a user's query
# after a failed attempt.

# Rules:

# - Preserve the original intent.
# - Do not add new facts.
# - Do not invent dates.
# - Do not invent customer information.
# - Do not invent account/card details.
# - Only address the failure feedback.

# Return ONLY JSON:

# {{
#     "revised_query": "...",
#     "issues_addressed": []
# }}
#                 """,
#             ),
#             (
#                 "human",
#                 """
# Original query:

# {original_query}


# Previous working query:

# {working_query}


# Failure stage:

# {failure_stage}


# Issues:

# {issues}
#                 """,
#             ),
#         ]
#     )

#     chain = prompt | llm

#     response = chain.invoke(
#         {
#             "original_query": state["original_query"],
#             "working_query": state["working_query"],
#             "failure_stage": feedback.get(
#                 "failure_stage",
#                 "unknown",
#             ),
#             "issues": feedback.get(
#                 "issues",
#                 [],
#             ),
#         }
#     )

#     import json

#     retry_query = json.loads(response.content)

#     retry = state.get(
#         "retry",
#         {
#             "count": 0,
#             "max_retries": MAX_RETRIES,
#         },
#     )

#     retry["count"] += 1

#     return {
#         "working_query": retry_query["revised_query"],
#         "retry_query": {
#             "revised_query": retry_query["revised_query"],
#             "retry_number": retry["count"],
#             "failure_stage": feedback.get("failure_stage"),
#             "issues_addressed": retry_query.get(
#                 "issues_addressed",
#                 [],
#             ),
#         },
#         "retry": retry,
#     }

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm


def retry_rephrase_node(
    state: dict[str, Any],
):
    """
    Rewrite query after evaluation failure.

    Purpose:
    - Improve retrieval/sql success on retry.
    - Preserve original intent.
    - Never drop parts of combined queries.
    """

    retry_count = state.get(
        "retry",
        {},
    ).get(
        "count",
        0,
    )

    max_retries = state.get(
        "retry",
        {},
    ).get(
        "max_retries",
        3,
    )

    decision = state.get("agent_decision", {})

    original_query = state.get("original_query", "")

    retry_feedback = state.get("retry_feedback", {})

    # ======================================================
    # IMPORTANT:
    # Combined queries contain multiple intents.
    #
    # Example:
    # "give credit card variants and send james details"
    #
    # Bad rewrite:
    # "give credit card variants"
    #
    # This removes analytics intent.
    #
    # Therefore preserve combined queries.
    # ======================================================

    if decision.get("query_type") == "combined":

        return {
            "working_query": original_query,
            "retry_query": {
                "original_query": original_query,
                "reason": ("Combined query preserved. " "All intents must remain."),
            },
            "retry": {
                "count": min(retry_count + 1, max_retries),
                "max_retries": max_retries,
            },
        }

    # ======================================================
    # Normal retry rewrite for FAQ / Analytics
    # ======================================================

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a query rewriting agent.

Your task:
Rewrite the user query to improve answer quality
after a previous attempt failed.

IMPORTANT:

If the original query is a conversation query, NEVER rewrite it
into an FAQ or analytics query.

Conversation queries must remain conversation queries.

Examples:

"who am i"
→ "who am i"

"what is my name"
→ "what is my name"

"who are you"
→ "who are you"

Do not transform these into requests for SQL, database evidence,
customer records, or FAQ evidence.

Rules:

- Preserve the original meaning.
- Do not add new information.
- Do not remove important entities.
- Keep names, dates, products, and identifiers.
- Make the query clearer.

Failure feedback:

{feedback}


Return ONLY the rewritten query.
""",
            ),
            (
                "human",
                """
Original user query:

{query}
""",
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "query": original_query,
            "feedback": retry_feedback.get("issues", []),
        }
    )

    rewritten_query = response.content.strip()

    return {
        "working_query": rewritten_query,
        "retry_query": {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "reason": "Retry query generated",
        },
        "retry": {
            "count": min(retry_count + 1, max_retries),
            "max_retries": max_retries,
        },
    }


def retry_exhausted_response(
    state: dict[str, Any],
):

    return {
        "final_response": {
            "query": state["original_query"],
            "answer": (
                "I couldn't produce a "
                "sufficiently reliable answer "
                "from the available information."
            ),
            "citations": "N/A",
            "page_no": "N/A",
            "document_name": "N/A",
            "sql_query_executed": None,
        }
    }
