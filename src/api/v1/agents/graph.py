from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.states.agent_state import AgentState
from src.agents.agents import agent_node

from src.agents.evaluators import (
    retrieval_evaluator_node,
    sql_evaluator_node,
    final_evaluator_node,
)

from src.agents.retry import retry_router, retry_rephrase_node, retry_exhausted_response

from src.services.retrieval_service import retrieve_documents
from src.services.rerank_service import rerank_documents
from src.services.sql_service import generate_sql
from src.services.sql_executor import execute_sql
from src.tools.sql_tools import validate_sql
from src.services.answer_service import answer_builder_node

# ============================================================
# Wrapper Nodes
# ============================================================


def retrieval_node(state: AgentState):

    decision = state["agent_decision"]

    result = retrieve_documents(
        query=state["working_query"],
        strategy=decision["retrieval_strategy"],
    )

    return {"retrieval_result": result}


def rerank_node(state: AgentState):

    retrieval = state["retrieval_result"]

    result = rerank_documents(
        query=state["working_query"],
        documents=retrieval["candidates"],
    )

    return {"rerank_result": result}


def nl2sql_node(state: AgentState):

    result = generate_sql(query=state["working_query"])

    return {"sql_generation": result}


def sql_validation_node(state: AgentState):

    sql = state["sql_generation"]["sql"]

    result = validate_sql(sql)

    return {"sql_validation": result}


def sql_execution_node(state: AgentState):

    validation = state["sql_validation"]

    if not validation["passed"]:

        return {
            "sql_execution": {
                "status": "error",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "error_code": "SQL_VALIDATION_FAILED",
                "error_message": ("SQL validation failed."),
            }
        }

    sql = state["sql_generation"]["sql"]

    result = execute_sql(sql)

    return {"sql_execution": result}


# ============================================================
# Routing Functions
# ============================================================


def route_after_agent(state: AgentState):

    decision = state["agent_decision"]

    if decision["needs_faq"] and decision["needs_analytics"]:
        return "combined"

    if decision["needs_faq"]:
        return "faq"

    if decision["needs_analytics"]:
        return "analytics"

    return "faq"


def route_combined(state: AgentState):
    """
    Fan-out helper.

    For combined requests both branches must execute.
    """

    return [
        "retrieval",
        "nl2sql",
    ]


def evidence_gate(state):

    decision = state["agent_decision"]

    if decision["needs_faq"]:

        if "retrieval_evaluation" not in state:
            return "wait"

        if not state["retrieval_evaluation"]["passed"]:
            return "retry"

    if decision["needs_analytics"]:

        if "sql_evaluation" not in state:
            return "wait"

        if not state["sql_evaluation"]["passed"]:
            return "retry"

    return "answer"


def route_final_evaluation(state: AgentState):

    evaluation = state["final_evaluation"]

    if evaluation["passed"]:
        return "end"

    return "retry"


# ============================================================
# Build Graph
# ============================================================


def build_graph():

    workflow = StateGraph(AgentState)

    # ---------------------------
    # Nodes
    # ---------------------------

    workflow.add_node(
        "agent",
        agent_node,
    )

    workflow.add_node(
        "retrieval",
        retrieval_node,
    )

    workflow.add_node(
        "rerank",
        rerank_node,
    )

    workflow.add_node(
        "retrieval_eval",
        retrieval_evaluator_node,
    )

    workflow.add_node(
        "nl2sql",
        nl2sql_node,
    )

    workflow.add_node(
        "sql_validate",
        sql_validation_node,
    )

    workflow.add_node(
        "sql_execute",
        sql_execution_node,
    )

    workflow.add_node(
        "sql_eval",
        sql_evaluator_node,
    )

    workflow.add_node(
        "evidence_gate",
        evidence_gate,
    )

    workflow.add_node(
        "answer_builder",
        answer_builder_node,
    )

    workflow.add_node(
        "final_eval",
        final_evaluator_node,
    )

    workflow.add_node(
        "retry_rephrase",
        retry_rephrase_node,
    )

    workflow.add_node(
        "retry_failed",
        retry_exhausted_response,
    )

    # ---------------------------
    # START
    # ---------------------------

    workflow.add_edge(START, "agent")

    # ---------------------------
    # Agent Routing
    # ---------------------------

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
    )

    # ---------------------------
    # FAQ branch
    # ---------------------------

    workflow.add_edge(
        "retrieval",
        "rerank",
    )

    workflow.add_edge(
        "rerank",
        "retrieval_eval",
    )

    workflow.add_edge(
        "retrieval_eval",
        "evidence_gate",
    )

    # ---------------------------
    # SQL branch
    # ---------------------------

    workflow.add_edge(
        "nl2sql",
        "sql_validate",
    )

    workflow.add_edge(
        "sql_validate",
        "sql_execute",
    )

    workflow.add_edge(
        "sql_execute",
        "sql_eval",
    )

    workflow.add_edge(
        "sql_eval",
        "evidence_gate",
    )

    # ---------------------------
    # Answer
    # ---------------------------

    workflow.add_conditional_edges(
        "evidence_gate",
        evidence_gate,
        {
            "answer": "answer_builder",
            "retry": "retry_rephrase",
        },
    )

    workflow.add_edge(
        "answer_builder",
        "final_eval",
    )

    workflow.add_conditional_edges(
        "final_eval",
        route_final_evaluation,
        {
            "end": END,
            "retry": "retry_rephrase",
        },
    )

    # ---------------------------
    # Retry
    # ---------------------------

    workflow.add_conditional_edges(
        "retry_rephrase",
        retry_router,
        {
            "retry": "agent",
            "exhausted": "retry_failed",
        },
    )

    workflow.add_edge(
        "retry_failed",
        END,
    )

    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
