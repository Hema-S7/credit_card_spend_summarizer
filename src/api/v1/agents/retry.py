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


def retry_rephrase_node(
    state: dict[str, Any],
):

    llm = get_llm()

    feedback = state.get(
        "retry_feedback",
        {},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a query refinement assistant.

Your job is to rewrite a user's query
after a failed attempt.

Rules:

- Preserve the original intent.
- Do not add new facts.
- Do not invent dates.
- Do not invent customer information.
- Do not invent account/card details.
- Only address the failure feedback.

Return ONLY JSON:

{
    "revised_query": "...",
    "issues_addressed": []
}
                """,
            ),
            (
                "human",
                """
Original query:

{original_query}


Previous working query:

{working_query}


Failure stage:

{failure_stage}


Issues:

{issues}
                """,
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "original_query": state["original_query"],
            "working_query": state["working_query"],
            "failure_stage": feedback.get(
                "failure_stage",
                "unknown",
            ),
            "issues": feedback.get(
                "issues",
                [],
            ),
        }
    )

    import json

    retry_query = json.loads(response.content)

    retry = state.get(
        "retry",
        {
            "count": 0,
            "max_retries": MAX_RETRIES,
        },
    )

    retry["count"] += 1

    return {
        "working_query": retry_query["revised_query"],
        "retry_query": {
            "revised_query": retry_query["revised_query"],
            "retry_number": retry["count"],
            "failure_stage": feedback.get("failure_stage"),
            "issues_addressed": retry_query.get(
                "issues_addressed",
                [],
            ),
        },
        "retry": retry,
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
