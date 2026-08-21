from langchain_openai import ChatOpenAI

from src.core.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def get_llm() -> ChatOpenAI:
    """
    Create and return the shared ChatOpenAI model
    used throughout the application.
    """

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not configured.")

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
