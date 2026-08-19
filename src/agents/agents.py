from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm


def contextualize_query_node(
    state: dict[str, Any],
):
    """
    Resolve references in the current query
    using conversation history.

    Keeps original_query unchanged and only
    updates working_query.
    """

    history = state.get("conversation_history") or []

    query = state["original_query"]

    # No previous history -> nothing to resolve
    if len(history) <= 1:
        return {"working_query": query}

    # Current user message was already added
    # by start_turn_node, so exclude it.
    previous_history = history[:-1]

    history_text = "\n".join(
        [f'{message["role"]}: {message["content"]}' for message in previous_history]
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You rewrite follow-up user questions into
self-contained questions using conversation history.

Rules:

- Preserve the user's intent.
- Resolve references such as:
  "it",
  "that card",
  "those cards",
  "the second one",
  "mentioned earlier",
  "previously mentioned".

- Use only information present in conversation history.
- Do not invent names, card IDs, dates, or facts.
- Do not answer the question.
- If the query is already self-contained,
  return it unchanged.

Return only the rewritten query.
                """,
            ),
            (
                "human",
                """
Conversation history:

{history}


Current query:

{query}
                """,
            ),
        ]
    )

    response = (prompt | llm).invoke(
        {
            "history": history_text,
            "query": query,
        }
    )

    rewritten_query = response.content.strip()

    return {"working_query": rewritten_query}


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

    retry_feedback = state.get("retry_feedback") or {}

    failure_stage = retry_feedback.get(
        "failure_stage",
        "None",
    )

    retry_issues = retry_feedback.get(
        "issues",
        [],
    )

    previous_strategy = (state.get("retrieval_result") or {}).get("strategy")

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

4. conversation

Questions that can be answered from the existing
conversation history and do not require FAQ retrieval
or customer analytics.

Examples:

"What is my name?"
"What did I ask earlier?"
"What card did we just discuss?"
"Can you repeat your previous answer?"

For conversation queries:

- needs_faq = false
- needs_analytics = false
- retrieval_strategy = null

5. unrelated

Requests outside the scope of credit card product knowledge,
customer spend analysis, or conversation follow-up.

Examples:

- "Can you get me a job?"
- "What's the weather today?"
- "Write Python code for me."
- "Book me a hotel."
- "Tell me a recipe."

For unrelated queries:

- needs_faq = false
- needs_analytics = false
- retrieval_strategy = null

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
 "query_type": "faq|analytics|combined|conversation",
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
