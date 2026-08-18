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

    retry_count = state.get(
        "retry",
        {},
    ).get(
        "count",
        0,
    )

    retry_feedback = state.get(
        "retry_feedback",
        {},
    )

    failure_stage = retry_feedback.get(
        "failure_stage",
        "None",
    )

    retry_issues = retry_feedback.get(
        "issues",
        [],
    )

    previous_strategy = state.get("retrieval_result", {}).get("strategy")

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

Retrieval strategy rules:

1. Use FTS when:
   - the question asks for one specific fact;
   - the query contains exact product names, terms, or identifiers;
   - keyword matching is likely sufficient.

2. Use semantic retrieval when:
   - the query is conceptual or paraphrased;
   - exact document terminology may differ from the user's wording.

3. Use hybrid retrieval when:
   - the query asks for multiple facts;
   - the query contains multiple conditions joined by "and";
   - the answer may require information from different document chunks;
   - both exact product names and semantic concepts are present.

Examples:

"What is the annual fee for NorthStar Classic?"
→ FTS

"Explain how foreign transaction charges work."
→ Semantic

"What is the annual fee for NorthStar Classic and
what cashback applies to groceries and gas?"
→ Hybrid

Retry rules:

- If retry_count is greater than 0, consider the previous
  retrieval failure when choosing the next retrieval strategy.

- If the previous retrieval strategy was FTS and it returned
  no relevant documents, do not choose FTS again.

- After an FTS retrieval failure, prefer HYBRID because it
  combines keyword and semantic retrieval.

- If HYBRID previously failed, SEMANTIC may be tried.

- Do not change the user's original intent just because
  retrieval failed.

Return ONLY JSON:

{{
 "query_type": "faq|analytics|combined",
 "needs_faq": true/false,
 "needs_analytics": true/false,
 "retrieval_strategy": "fts|semantic|hybrid|null"
}}

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


Retry count:

{retry_count}


Previous retrieval strategy:

{previous_strategy}


Previous failure stage:

{failure_stage}


Previous failure issues:

{retry_issues}
    """,
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "query": state["working_query"],
            "history": state.get(
                "conversation_history",
                [],
            ),
            "retry_count": retry_count,
            "previous_strategy": (previous_strategy or "None"),
            "failure_stage": failure_stage,
            "retry_issues": retry_issues,
        }
    )

    import json

    decision = json.loads(response.content)

    return {"agent_decision": decision}
