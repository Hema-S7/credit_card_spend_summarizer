from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.states.agent_state import AgentState
from src.agents.agents import agent_node, contextualize_query_node

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
from src.services.conversation_service import (
    start_turn_node,
    save_assistant_message_node,
)

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


def route_after_agent(
    state: AgentState,
):

    decision = state["agent_decision"]

    if decision["query_type"] == "conversation":
        return "conversation"

    if decision["query_type"] == "unrelated":
        return "unrelated"

    if decision["needs_faq"] and decision["needs_analytics"]:
        return "combined"

    if decision["needs_faq"]:
        return "faq"

    if decision["needs_analytics"]:
        return "analytics"

    return "unrelated"


def route_combined(state: AgentState):
    """
    Fan-out helper.

    For combined requests both branches must execute.
    """

    return [
        "retrieval",
        "nl2sql",
    ]


def evidence_gate_node(
    state: AgentState,
):
    """
    Synchronization/gate node.

    A LangGraph node must return a dict.
    No state update is required here.
    """
    return {}


def route_after_evidence_gate(
    state: AgentState,
):
    """
    Decide whether validated evidence is ready
    for answer generation or retry is required.
    """

    decision = state["agent_decision"]

    # FAQ evidence required
    if decision["needs_faq"]:

        retrieval_eval = state.get("retrieval_evaluation")

        if not retrieval_eval or not retrieval_eval["passed"]:
            return "retry"

    # Analytics evidence required
    if decision["needs_analytics"]:

        sql_eval = state.get("sql_evaluation")

        if not sql_eval or not sql_eval["passed"]:
            return "retry"

    return "answer"


def retry_check_node(
    state: AgentState,
):

    issues = []
    failure_stages = []

    retrieval_eval = state.get("retrieval_evaluation")

    if retrieval_eval and not retrieval_eval["passed"]:

        issues.extend(retrieval_eval.get("issues", []))

        failure_stages.append("retrieval_evaluation")

    sql_eval = state.get("sql_evaluation")

    if sql_eval and not sql_eval["passed"]:

        issues.extend(sql_eval.get("issues", []))

        failure_stages.append("sql_evaluation")

    final_eval = state.get("final_evaluation")

    if final_eval and not final_eval["passed"]:

        issues.extend(final_eval.get("issues", []))

        failure_stages.append("final_evaluation")

    return {
        "retry_feedback": {
            "failure_stage": (
                ", ".join(failure_stages) if failure_stages else "unknown"
            ),
            "error_code": None,
            "issues": issues,
        }
    }


def route_final_evaluation(state: AgentState):

    evaluation = state["final_evaluation"]

    if evaluation["passed"]:
        return "end"

    return "retry"


def combined_start_node(state: AgentState):
    return {}


def faq_complete_node(state: AgentState):
    return {}


def sql_complete_node(state: AgentState):
    return {}


def unrelated_response_node(
    state: dict,
):

    return {
        "answer_draft": {
            "query": state["original_query"],
            "answer": (
                "I can help with credit card product information "
                "and customer spend analysis. "
                "This request is outside the scope of this assistant."
            ),
            "citations": "N/A",
            "page_no": "N/A",
            "document_name": "N/A",
            "sql_query_executed": None,
        }
    }


def route_after_retrieval_eval(
    state: AgentState,
):
    decision = state["agent_decision"]

    if decision["query_type"] == "combined":
        return "combined"

    return "single"


def route_after_sql_eval(
    state: AgentState,
):
    decision = state["agent_decision"]

    if decision["query_type"] == "combined":
        return "combined"

    return "single"


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
        evidence_gate_node,
    )

    workflow.add_node(
        "answer_builder",
        answer_builder_node,
    )

    workflow.add_node(
        "contextualize_query",
        contextualize_query_node,
    )

    workflow.add_node(
        "final_eval",
        final_evaluator_node,
    )
    workflow.add_node(
        "retry_check",
        retry_check_node,
    )
    workflow.add_node(
        "retry_rephrase",
        retry_rephrase_node,
    )

    workflow.add_node(
        "retry_failed",
        retry_exhausted_response,
    )

    workflow.add_node(
        "combined_start",
        combined_start_node,
    )

    workflow.add_node(
        "faq_complete",
        faq_complete_node,
    )

    workflow.add_node(
        "sql_complete",
        sql_complete_node,
    )

    workflow.add_node(
        "start_turn",
        start_turn_node,
    )

    workflow.add_node(
        "save_assistant",
        save_assistant_message_node,
    )

    workflow.add_node(
        "unrelated_response",
        unrelated_response_node,
    )

    # ---------------------------
    # START
    # ---------------------------

    workflow.add_edge(
        START,
        "start_turn",
    )

    workflow.add_edge(
        "start_turn",
        "contextualize_query",
    )

    workflow.add_edge(
        "contextualize_query",
        "agent",
    )

    # ---------------------------
    # Agent Routing
    # ---------------------------

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "faq": "retrieval",
            "analytics": "nl2sql",
            "combined": "combined_start",
            "conversation": "answer_builder",
            "unrelated": "unrelated_response",
        },
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

    workflow.add_conditional_edges(
        "retrieval_eval",
        route_after_retrieval_eval,
        {
            "single": "evidence_gate",
            "combined": "faq_complete",
        },
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

    workflow.add_conditional_edges(
        "sql_eval",
        route_after_sql_eval,
        {
            "single": "evidence_gate",
            "combined": "sql_complete",
        },
    )

    # ---------------------------
    # Combined branch
    # ---------------------------
    workflow.add_edge(
        "combined_start",
        "retrieval",
    )

    workflow.add_edge(
        "combined_start",
        "nl2sql",
    )

    workflow.add_edge(
        [
            "faq_complete",
            "sql_complete",
        ],
        "evidence_gate",
    )

    # ---------------------------
    # Answer
    # ---------------------------

    workflow.add_conditional_edges(
        "evidence_gate",
        route_after_evidence_gate,
        {
            "answer": "answer_builder",
            "retry": "retry_check",
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
            "end": "save_assistant",
            "retry": "retry_check",
        },
    )

    # ---------------------------
    # Retry
    # ---------------------------
    workflow.add_conditional_edges(
        "retry_check",
        retry_router,
        {
            "retry": "retry_rephrase",
            "exhausted": "retry_failed",
        },
    )

    workflow.add_edge(
        "retry_rephrase",
        "agent",
    )

    workflow.add_edge(
        "unrelated_response",
        "save_assistant",
    )

    workflow.add_edge(
        "retry_failed",
        "save_assistant",
    )

    workflow.add_edge(
        "save_assistant",
        END,
    )

    checkpointer = MemorySaver()

    return_flow = workflow.compile(checkpointer=checkpointer)

    # generate and save the graph visualization
    graph_image = return_flow.get_graph().draw_mermaid_png()
    with open("test_dig.png", "wb") as f:
        f.write(graph_image)

    return return_flow
