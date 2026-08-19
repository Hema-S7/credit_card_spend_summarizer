from typing import Any

from langchain_core.prompts import (
    ChatPromptTemplate,
)
from pydantic import BaseModel, Field

from src.core.llm import get_llm


class AnswerDraft(BaseModel):

    answer: str = Field(description=("Grounded answer to the user's question"))

    used_chunk_ids: list[str] = Field(
        default_factory=list,
        description=("Chunk IDs actually used to " "construct the answer"),
    )


def answer_builder_node(
    state: dict[str, Any],
):

    llm = get_llm()

    structured_llm = llm.with_structured_output(AnswerDraft)

    context_parts = []

    faq_documents = []

    # -----------------------------------------
    # Conversation evidence
    # -----------------------------------------

    history = state.get("conversation_history") or []

    # start_turn_node already added the
    # current user question as the last message.
    # We only want previous conversation as evidence.

    previous_history = history

    if (
        previous_history
        and previous_history[-1]["role"] == "user"
        and previous_history[-1]["content"] == state["original_query"]
    ):

        previous_history = previous_history[:-1]

    if previous_history:

        history_text = "\n".join(
            [f'{message["role"]}: {message["content"]}' for message in previous_history]
        )

        context_parts.append(f"""
    Conversation History:

    {history_text}
    """)

    # -----------------------------------------
    # FAQ evidence
    # -----------------------------------------

    rerank_result = state.get("rerank_result")

    if rerank_result:

        for doc in rerank_result.get(
            "documents",
            [],
        ):

            metadata = doc.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                "unknown",
            )

            page = metadata.get(
                "page",
                "unknown",
            )

            chunk_id = metadata.get(
                "chunk_id",
                "unknown",
            )

            faq_documents.append(doc)

            context_parts.append(f"""
Chunk ID: {chunk_id}
Source: {source}
Page: {page}

Content:
{doc["content"]}
""")

    # -----------------------------------------
    # SQL evidence
    # -----------------------------------------

    sql_execution = state.get("sql_execution")

    sql_query = None

    if sql_execution:

        sql_query = state.get(
            "sql_generation",
            {},
        ).get("sql")

        context_parts.append(f"""
Analytics Result:

Columns:
{sql_execution.get("columns")}

Rows:
{sql_execution.get("rows")}
""")

    context = "\n\n".join(context_parts)

    # -----------------------------------------
    # Answer prompt
    # -----------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a credit card spend
summarization assistant.

Answer the user question using ONLY
the provided evidence.

Rules:

- Do not invent information.
- Do not assume missing data.
- Do not use outside knowledge.

For FAQ information:
Use only the provided documents.

For analytics:
Use only the SQL results.

For combined questions:
Answer both parts clearly.

For conversation questions:
- Use only the provided conversation history.
- Do not use outside knowledge.
- Do not invent information that was not previously
  provided by the user or assistant.
- If the requested information does not exist in
  conversation history, say that it is not available.

FAQ evidence tracking:

- Every FAQ evidence block has a Chunk ID.

- When you use factual information from
  FAQ evidence, include that Chunk ID
  in used_chunk_ids.

- Include only Chunk IDs that actually
  contributed information to the answer.

- Do not include a chunk just because
  it was retrieved.

- Never invent a Chunk ID.

- used_chunk_ids must contain only
  Chunk IDs provided in the evidence.

- If no FAQ evidence is used,
  return an empty list.

For multi-part questions:
Different parts of the answer may use
different chunks. Include every chunk
that actually contributed to the answer.


SQL:

Use SQL evidence only when analytics
information is required.
                    """,
            ),
            (
                "human",
                """
Question:

{query}


Evidence:

{context}
                    """,
            ),
        ]
    )

    chain = prompt | structured_llm

    response = chain.invoke(
        {
            "query": state["original_query"],
            "context": context,
        }
    )

    # -----------------------------------------
    # Determine FAQ chunks actually used
    # -----------------------------------------

    used_chunk_ids = {str(chunk_id) for chunk_id in response.used_chunk_ids}

    used_documents = []

    for doc in faq_documents:

        metadata = doc.get(
            "metadata",
            {},
        )

        chunk_id = str(
            metadata.get(
                "chunk_id",
                "",
            )
        )

        if chunk_id in used_chunk_ids:

            used_documents.append(doc)

    # -----------------------------------------
    # Build citation metadata
    # -----------------------------------------

    citations = []
    pages = []
    document_names = []

    for doc in used_documents:

        metadata = doc.get(
            "metadata",
            {},
        )

        source = metadata.get("source")

        page = metadata.get("page")

        if source:

            if source not in citations:
                citations.append(source)

            if source not in document_names:
                document_names.append(source)

        if page is not None:

            page_string = str(page)

            if page_string not in pages:
                pages.append(page_string)

    # -----------------------------------------
    # Final AIResponse-compatible structure
    # -----------------------------------------

    result = {
        "query": state["original_query"],
        "answer": response.answer,
        "citations": ("; ".join(citations) if citations else "N/A"),
        "page_no": ("; ".join(pages) if pages else "N/A"),
        "document_name": (
            "; ".join(document_names)
            if document_names
            else ("agentic_rag_db" if sql_execution else "N/A")
        ),
        "sql_query_executed": (sql_query),
    }

    return {"answer_draft": result}
