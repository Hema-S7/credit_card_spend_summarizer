from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm


def agent_node(
    state: dict[str, Any],
):
    """
    Decide workflow routing.

    Determines:
    - FAQ required?
    - Analytics required?
    - Combined?
    - Retrieval strategy?
    """

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a routing agent for a credit card spend
summarization assistant.

Classify the user query.

Possible query types:

1. faq
Questions about:
- card benefits
- policies
- fees
- features
- general credit card information

2. analytics
Questions requiring customer data analysis:
- spending
- transactions
- summaries
- trends
- category analysis

3. combined
Questions requiring both:
- FAQ knowledge
- customer transaction data


For FAQ queries also decide retrieval strategy:

fts:
- exact terms
- names
- policy keywords

semantic:
- conceptual questions
- meaning-based questions

hybrid:
- when both keyword and semantic matching help


Return ONLY JSON:

{
 "query_type": "faq|analytics|combined",
 "needs_faq": true/false,
 "needs_analytics": true/false,
 "retrieval_strategy": "fts|semantic|hybrid|null"
}

Do not answer the user question.
Only classify.
                """,
            ),
            (
                "human",
                """
Conversation history:

{history}


Current user query:

{query}
                """,
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "query": state["working_query"],
            "history": state.get("conversation_history", []),
        }
    )

    import json

    decision = json.loads(response.content)

    return {"agent_decision": decision}
