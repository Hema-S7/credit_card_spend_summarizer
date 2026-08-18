from typing import Any


def add_message(
    state: dict[str, Any],
    role: str,
    content: str,
):

    history = state.get(
        "conversation_history",
        [],
    )

    history.append(
        {
            "role": role,
            "content": content,
        }
    )

    return {"conversation_history": history}
