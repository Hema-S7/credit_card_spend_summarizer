from typing import Any


def append_message(
    state: dict[str, Any],
    role: str,
    content: str,
):
    """
    Return conversation history with
    one new message appended.
    """

    history = list(
        state.get(
            "conversation_history",
            [],
        )
        or []
    )

    history.append(
        {
            "role": role,
            "content": content,
        }
    )

    return history


def start_turn_node(
    state: dict[str, Any],
):
    """
    Start a new user turn.

    - Store the new user message
    - Clear temporary state from the previous turn
    """

    query = state["original_query"]

    return {
        "conversation_history": append_message(
            state,
            role="user",
            content=query,
        ),
        # Clear previous-turn workflow data
        "agent_decision": None,
        "retrieval_result": None,
        "rerank_result": None,
        "retrieval_evaluation": None,
        "sql_generation": None,
        "sql_validation": None,
        "sql_execution": None,
        "sql_evaluation": None,
        "answer_draft": None,
        "final_evaluation": None,
        "retry_feedback": None,
        "retry_query": None,
        "workflow_error": None,
        "final_response": None,
    }


def save_assistant_message_node(
    state: dict[str, Any],
):
    """
    Save the validated assistant response
    into conversation history.
    """

    response = state.get("final_response") or state.get("answer_draft")

    if not response:
        return {}

    answer = response.get("answer")

    if not answer:
        return {}

    return {
        "conversation_history": append_message(
            state,
            role="assistant",
            content=answer,
        )
    }
