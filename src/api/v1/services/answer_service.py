from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.api.v1.schemas.query_schema import (
    AIResponse,
)

from src.core.llm import get_llm


def answer_builder_node(
    state: dict[str, Any],
):
    print("Building answer from evidence...")
    """
        Generate final structured response.

        Uses only:
        - validated FAQ documents
        - validated SQL results

 
    """

    llm = get_llm()

    structured_llm = llm.with_structured_output(AIResponse)

    history = state.get("conversation_history", [])

    history_text = "\n\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]
    )

    context_parts = []

    citations = []
    pages = []
    documents = []

    # -----------------------------------------
    # FAQ evidence
    # -----------------------------------------

    rerank_result = state.get("rerank_result")

    if rerank_result:

        for doc in rerank_result.get(
            "documents",
            [],
        ):

            metadata = doc.get("metadata", {})

            source = metadata.get(
                "source",
                "unknown",
            )

            page = metadata.get(
                "page",
                "unknown",
            )

            context_parts.append(f"""
Source: {source}
Page: {page}

{doc["content"]}
""")

            citations.append(source)

            pages.append(str(page))

            documents.append(source)

    # -----------------------------------------
    # SQL evidence
    # -----------------------------------------

    sql_execution = state.get("sql_execution")

    sql_query = None

    if sql_execution:

        sql_query = state.get("sql_generation", {}).get("sql")

        context_parts.append(f"""
Analytics Result:

Columns:
{sql_execution.get("columns")}


Rows:
{sql_execution.get("rows")}
""")

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a credit card spend summarization assistant.

Answer the user question using ONLY the
provided evidence.

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

Citations:
- Include every FAQ document used.
- Include page numbers of cite chunks actually used to answer the question.
- If no FAQ documents were used,
  use "N/A".

SQL:
- Include the SQL query only when analytics
  data was used.
- Otherwise set it to null.

Conversation history:
- Use only for resolving references like:
  "it", "that card", "previous one", "what about fees".
- use conversation history as a source for replying for conversational questions.
- Facts do not need to come only from evidence.

""",
            ),
            (
                "human",
                """
Conversation history:

{history}

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
            "query": state["working_query"],  # chnage to workingquery
            "history": history_text,
            "context": context,
        }
    )

    result = response.model_dump()

    # -----------------------------------------
    # Populate citations
    # -----------------------------------------

    if citations:

        result["citations"] = "; ".join(sorted(set(citations)))

        result["page_no"] = "; ".join(pages)

        result["document_name"] = "; ".join(sorted(set(documents)))

    else:

        result["citations"] = "N/A"
        result["page_no"] = "N/A"
        result["document_name"] = "agentic_rag_db" if sql_execution else "N/A"

    result["sql_query_executed"] = sql_query

    return {"answer_draft": result}
