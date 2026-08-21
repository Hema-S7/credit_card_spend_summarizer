from typing import Any

from src.core.config import RETRIEVAL_THRESHOLD
from src.core.llm import get_llm


def build_faq_context(
    state: dict[str, Any],
):

    rerank_result = state.get("rerank_result")

    if not rerank_result:
        return "No FAQ evidence"

    return "\n\n".join(
        [
            doc["content"]
            for doc in rerank_result.get(
                "documents",
                [],
            )
        ]
    )


def build_sql_context(
    state: dict[str, Any],
):

    sql_execution = state.get("sql_execution")

    if not sql_execution:
        return "No SQL evidence"

    return str(
        sql_execution.get(
            "rows",
            [],
        )
    )


def judge_final_answer(
    query: str,
    answer: str,
    faq_context: str,
    sql_context: str,
    query_type: str,
):

    llm = get_llm()

    prompt = f"""
You are the final quality evaluator
for a credit card spend summarization assistant.


Original user question:

{query}


Query type:

{query_type}


Generated answer:

{answer}


FAQ evidence:

{faq_context}


SQL evidence:

{sql_context}


Evaluation rules:


1. Correct:
- Is the answer factually correct?
- Are claims supported by FAQ or SQL evidence?


2. Complete:
- Does the answer address every part of the user question?


IMPORTANT:

If query_type is "combined":

The user asked two different things:

A) FAQ part:
- Must be answered using FAQ evidence.

B) Analytics part:
- Must be answered using SQL evidence.


A combined answer is incomplete if:
- FAQ information is missing
OR
- SQL/customer information is missing.


3. Grounded:
- Every factual statement must come from FAQ evidence or SQL evidence.
- Do not allow invented customer information.


Return ONLY JSON:

{{
 "correct": true/false,
 "complete": true/false,
 "grounded": true/false,
 "score": 0.0-1.0,
 "issues": []
}}
"""

    response = llm.invoke(prompt)

    import json

    return json.loads(response.content)


def final_evaluator_node(
    state: dict[str, Any],
):
    print("FINAL EVALUATOR NODE")
    answer = state.get("answer_draft")

    if not answer:

        return {
            "final_evaluation": {
                "correct": False,
                "complete": False,
                "grounded": False,
                "score": 0.0,
                "passed": False,
                "retry_required": True,
                "issues": ["No answer draft available."],
            }
        }

    faq_context = build_faq_context(state)

    sql_context = build_sql_context(state)

    evaluation = judge_final_answer(
        query=state["working_query"],  # change to working
        # IMPORTANT
        # send only answer text
        answer=answer.get(
            "response",
            "",
        ),
        faq_context=faq_context,
        sql_context=sql_context,
        query_type=state.get("agent_decision", {}).get(
            "query_type",
            "unknown",
        ),
    )
    print("FINAL EVAL:", evaluation)
    passed = evaluation["correct"] and evaluation["complete"] and evaluation["grounded"]

    return {
        "final_evaluation": {
            **evaluation,
            "passed": passed,
            "retry_required": not passed,
        }
    }


def retrieval_evaluator_node(
    state: dict[str, Any],
):
    """
    Evaluate retrieved FAQ documents.

    Checks:
    1. similarity/relevance threshold
    2. LLM relevance judgement
    """

    rerank_result = state.get("rerank_result")

    query = state.get("working_query")

    if not rerank_result:

        return {
            "retrieval_evaluation": {
                "threshold_passed": False,
                "llm_relevance_passed": False,
                "threshold_score": None,
                "llm_relevance_score": None,
                "passed": False,
                "retry_required": True,
                "issues": ["No reranked documents available."],
            }
        }

    documents = rerank_result["documents"]

    # if not documents:

    #     return {
    #         "retrieval_evaluation": {
    #             "threshold_passed": False,
    #             "llm_relevance_passed": False,
    #             "threshold_score": 0,
    #             "llm_relevance_score": 0,
    #             "passed": False,
    #             "retry_required": True,
    #             "issues": ["No relevant documents retrieved."],
    #         }
    #     }

    if not documents:

        query_type = state.get("agent_decision", {}).get(
            "query_type",
            "unknown",
        )

        # No FAQ documents are acceptable for analytics-only queries.
        if query_type == "analytics":

            return {
                "retrieval_evaluation": {
                    "threshold_passed": True,
                    "llm_relevance_passed": True,
                    "threshold_score": 0,
                    "llm_relevance_score": 0,
                    "passed": True,
                    "retry_required": False,
                    "issues": [],
                }
            }

        return {
            "retrieval_evaluation": {
                "threshold_passed": False,
                "llm_relevance_passed": False,
                "threshold_score": 0,
                "llm_relevance_score": 0,
                "passed": False,
                "retry_required": True,
                "issues": ["No relevant documents retrieved."],
            }
        }

    # -----------------------------------------
    # 1. Similarity threshold check
    # -----------------------------------------

    rerank_scores = [doc["rerank_score"] for doc in documents]

    best_score = max(rerank_scores)

    threshold_passed = best_score >= RETRIEVAL_THRESHOLD

    # -----------------------------------------
    # 2. LLM relevance judge
    # -----------------------------------------

    llm_result = judge_relevance(
        query=query,
        documents=documents,
    )

    llm_passed = llm_result["passed"]

    passed = threshold_passed or llm_passed  # changed and to or

    issues = []

    if not threshold_passed:

        issues.append("Low similarity score but LLM confirmed relevance..")

    if not llm_passed:

        issues.append("LLM judge found retrieved documents insufficient.")

    return {
        "retrieval_evaluation": {
            "threshold_passed": threshold_passed,
            "llm_relevance_passed": llm_passed,
            "threshold_score": best_score,
            "llm_relevance_score": (llm_result["score"]),
            "passed": passed,
            "retry_required": not passed,
            "issues": issues,
        }
    }


def judge_relevance(
    query: str,
    documents: list[dict[str, Any]],
):

    llm = get_llm()

    context = "\n\n".join([f"""
            Document {i+1}:

            {doc["content"]}
            """ for i, doc in enumerate(documents)])

    prompt = f"""
You are a retrieval quality evaluator.

User query:

{query}


Retrieved documents:

{context}


Determine whether these documents contain
sufficient information to answer the user query.


Important evaluation rules:

- A retrieval result can still be sufficient even when
  the user's terminology does not exactly match the
  terminology used in the documents.

- Do not require exact keyword matches if the retrieved
  documents clearly contain equivalent or closely related
  information.

- Example:
  The user may ask for "cashback", while the document may
  provide a "reward points rate".

- If the documents contain enough information to:
  1. answer the supported part of the user's question, and
  2. clearly state that another requested concept is not
     explicitly available in the evidence,

  then the retrieval should be considered sufficient.

- Do not mark the retrieval as insufficient merely because
  the exact wording used by the user is absent.

- For multi-part questions, consider the retrieved documents
  together. Different parts of the answer may come from
  different documents/chunks.

- Return NO only when the retrieved evidence is genuinely
  insufficient to provide a grounded and useful answer.


Return ONLY:

YES or NO
"""

    response = llm.invoke(prompt)

    answer = response.content.strip().upper()

    passed = answer == "YES"

    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
    }


from typing import Any

from src.core.llm import get_llm


def sql_evaluator_node(
    state: dict[str, Any],
):
    """
    Evaluate SQL correctness and result quality.

    Checks:
    - correctness of SQL logic
    - completeness of returned result
    - grounding of result
    """

    query = state.get("working_query")

    sql_generation = state.get("sql_generation")

    sql_execution = state.get("sql_execution")

    if not sql_generation or not sql_execution:

        return {
            "sql_evaluation": {
                "correct": False,
                "complete": False,
                "grounded": False,
                "score": 0.0,
                "passed": False,
                "retry_required": True,
                "issues": ["SQL generation or execution result missing."],
            }
        }

    if sql_execution["status"] in [
        "error",
        "timeout",
    ]:

        return {
            "sql_evaluation": {
                "correct": False,
                "complete": False,
                "grounded": False,
                "score": 0.0,
                "passed": False,
                "retry_required": True,
                "issues": ["SQL execution failed."],
            }
        }

    query_type = state.get("agent_decision", {}).get(
        "query_type",
        "unknown",
    )

    llm_result = judge_sql_result(
        query=query,
        sql=sql_generation["sql"],
        result=sql_execution,
        query_type=query_type,
    )

    passed = llm_result["correct"] and llm_result["grounded"]

    return {
        "sql_evaluation": {
            **llm_result,
            "passed": passed,
            "retry_required": not passed,
        }
    }


def judge_sql_result(
    query: str,
    sql: str,
    result: dict[str, Any],
    query_type: str,
):

    llm = get_llm()

    prompt = f"""
You are a SQL quality evaluator.


Evaluate whether the SQL query and its result
correctly answer the ANALYTICS portion of the user's question.

Query type:

{query_type}

IMPORTANT:

If query_type is "combined", the user's question contains
multiple parts.

SQL is responsible ONLY for the analytics/customer-data portion.

Do NOT require SQL to answer:
- FAQ questions
- product information
- reward rules
- cashback/benefit explanations
- general credit-card information

Those parts are handled separately using FAQ evidence.

For a combined query, SQL should PASS if it correctly
retrieves the customer/analytics information requested,
even if the SQL result does not contain the FAQ information.

User question:

{query}


SQL executed:

{sql}


SQL result:

{result}


Evaluate:

1. Correct:
Does the SQL logic correctly answer the question?

2. Complete:
Does the result contain all information required?

3. Grounded:
Is the answer supported by the SQL result?

Time-period evaluation rules:

When evaluating relative time queries such as:
- "this month"
- "last month"
- "current month"
- "previous month"

check whether the SQL selected an appropriate
time reference for the available dataset.

Do not automatically consider CURRENT_DATE correct.

If the dataset contains historical or static customer data,
relative periods should normally refer to the latest
available relevant data period unless the user explicitly
asks about the actual current calendar date.

Verify that:
- the selected periods match the user's intended comparison;
- the correct month/date columns were used;
- the returned periods actually contain the relevant data;
- a zero result is not caused merely by choosing calendar
  months outside the available dataset.



Return ONLY JSON:

{{
    "correct": true/false,
    "complete": true/false,
    "grounded": true/false,
    "score": 0.0-1.0,
    "issues": []
}}
"""

    response = llm.invoke(prompt)

    import json

    evaluation = json.loads(response.content)

    return evaluation
