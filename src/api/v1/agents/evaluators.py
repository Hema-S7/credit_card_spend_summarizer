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
    answer: dict,
    faq_context: str,
    sql_context: str,
):

    llm = get_llm()

    prompt = f"""
You are the final quality evaluator
for a credit card spend summarization assistant.


Original user question:

{query}


Generated answer:

{answer}


FAQ evidence:

{faq_context}


SQL evidence:

{sql_context}


Evaluate:


1. Correct:
Is the answer factually correct
based on the evidence?


2. Complete:
Does it answer every part
of the user's question?


3. Grounded:
Are all factual claims supported
by FAQ or SQL evidence?


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
    """
    Final quality gate.

    Evaluates:
    - correctness
    - completeness
    - grounding

    against original user intent.
    """

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
        query=state["original_query"],
        answer=answer,
        faq_context=faq_context,
        sql_context=sql_context,
    )

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

    if not documents:

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

    passed = threshold_passed and llm_passed

    issues = []

    if not threshold_passed:

        issues.append("Document relevance score below threshold.")

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

    llm_result = judge_sql_result(
        query=query,
        sql=sql_generation["sql"],
        result=sql_execution,
    )

    passed = llm_result["correct"] and llm_result["complete"] and llm_result["grounded"]

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
):

    llm = get_llm()

    prompt = f"""
You are a SQL quality evaluator.

Evaluate whether the SQL query and its result
correctly answer the user's question.

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
