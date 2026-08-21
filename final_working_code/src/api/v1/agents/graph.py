from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.api.v1.states.agent_state import AgentState
from src.api.v1.agents.agents import agent_node

from src.api.v1.agents.evaluators import (
    retrieval_evaluator_node,
    sql_evaluator_node,
    final_evaluator_node,
)

from src.api.v1.agents.retry import (
    retry_router,
    retry_rephrase_node,
    retry_exhausted_response,
)

from src.api.v1.services.retrieval_service import retrieve_documents
from src.api.v1.services.rerank_service import rerank_documents
from src.api.v1.services.sql_service import generate_sql
from src.api.v1.services.sql_executor import execute_sql
from src.api.v1.tools.sql_tools import validate_sql
from src.api.v1.services.answer_service import answer_builder_node
from typing import Generator
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from src.core.llm import get_llm

# ============================================================
# Wrapper Nodes
# ============================================================


def add_user_message_node(
    state: AgentState,
):

    history = state.get(
        "conversation_history",
        [],
    )

    history.append(
        {
            "role": "user",
            "content": state["original_query"],
        }
    )

    return {"conversation_history": history}


def add_ai_message_node(state: AgentState):

    answer = state.get("answer_draft", {}).get("response", "")

    history = state.get("conversation_history", [])

    history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return {"conversation_history": history}


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


def conversation_node(state: AgentState) -> dict[str, Any]:
    """
    Handles normal conversation such as greetings,
    acknowledgements, and general assistant interaction.

    Does not use retrieval or SQL.
    """

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a friendly AI assistant for a credit card spend
summarization application.

Handle simple conversational messages.

Conversation includes:
- greetings
- thanks
- acknowledgements
- asking who you are
- asking what you can do
- asking the user's previous provided information
- asking user's name
- asking about previous messages
- casual questions

===============================
If the user asks something unrelated to credit cards,
do not answer that question.

Politely redirect:

"Please ask a question related to NORTHSTAR Bank credit cards,
such as card benefits, fees, rewards, transactions, or spending."


Examples:

User:
"What is the weather today?"

Response:
"Please ask a question related to NORTHSTAR Bank credit cards,
such as card benefits, fees, rewards, transactions, or spending."

User:
"Tell me about loans"

Response:
"Please ask a question related to NORTHSTAR Bank credit cards,
such as card benefits, fees, rewards, transactions, or spending."

================================

Rules:
- Keep responses short and helpful.
- Do not provide credit card policy details unless asked.
- Do not invent banking information.
- Do not call retrieval or analytics tools.

- Use conversation history when replying.
- If the user previously provided their name, you may use it.
- Do not invent names or personal details.
- Only use information explicitly present in conversation history.
- Do not output routing JSON.

-If the user previously provided personal information
like name, preferences, etc., use that information.

-Do not say you don't know something if it exists
in conversation history.

- If the user asks to see, provide, reveal, display, reproduce, or explain
the SQL query used internally, then politely deny the request.

Examples:
- "show sql"
- "give sql"
- "provide the SQL query"
- "show me the query"
- "what SQL did you use?"
- "give me the SQL you executed"
- "display the SQL"
- "what query generated this answer?

Keep responses short.
""",
            ),
            (
                "human",
                """
Conversation history:

{history}


Current user message:

{query}

""",
            ),
        ]
    )

    chain = prompt | llm
    history = state.get(
        "conversation_history",
        []
    )

    history_text = "\n".join(
        [
            f"{m['role']}: {m['content']}"
            for m in history#[-6:]
        ]
    )

    response = chain.invoke(
        {
            "query": state["working_query"],
            "history": history_text,
        }
    )

    return {
        "answer_draft": {
            "query": state["working_query"],
            "history": history_text,
            "response": response.content,
            "policy_citations": "N/A",
            "document_name": "N/A",
            "page_no": "N/A",
            "sql_query_executed": None,
            "citations": "N/A",
        }
    }


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

    if decision["query_type"] == "conversation":
        return "conversation"

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


def evidence_gate_node(
    state: AgentState,
):
    """
    Synchronization/gate node.

    A LangGraph node must return a dict.
    No state update is required here.
    """
    print("EVIDENCE GATE")

    print("retrieval evaluation:", state.get("retrieval_evaluation"))

    print("sql evaluation:", state.get("sql_evaluation"))
    return {}


def combined_evidence_gate_node(
    state: AgentState,
):
    print("COMBINED EVIDENCE GATE")

    faq_done = state.get(
        "faq_completed",
        False,
    )

    sql_done = state.get(
        "sql_completed",
        False,
    )

    # print("FAQ DONE:", faq_done)
    # print("SQL DONE:", sql_done)

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
    # print("FAQ COMPLETE NODE")

    return {"faq_completed": True}


def sql_complete_node(state: AgentState):
    # print("SQL COMPLETE NODE")

    return {"sql_completed": True}


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
    "add_user_message",
    add_user_message_node,
    )

    workflow.add_node(
        "add_ai_message",
        add_ai_message_node,
    )

    workflow.add_node(
        "agent",
        agent_node,
    )
    workflow.add_node(
        "conversation",
        conversation_node,
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
        "combined_evidence_gate",
        combined_evidence_gate_node,
    )
    # ---------------------------
    # START
    # ---------------------------

    workflow.add_edge(
    START,
    "add_user_message"
    )

    workflow.add_edge(
        "add_user_message",
        "agent"
    )

    # ---------------------------
    # Agent Routing
    # ---------------------------

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "conversation": "conversation",
            "faq": "retrieval",
            "analytics": "nl2sql",
            "combined": "combined_start",
        },
    )

    # ---------------------------
    # FAQ branch
    # ---------------------------
    workflow.add_edge(
        "conversation",
        "final_eval",
    )

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

    # workflow.add_edge(
    #     [
    #         "faq_complete",
    #         "sql_complete",
    #     ],
    #     "evidence_gate",
    # )
    workflow.add_edge(
        "faq_complete",
        "combined_evidence_gate",
    )

    workflow.add_edge(
        "sql_complete",
        "combined_evidence_gate",
    )

    workflow.add_edge(
        "combined_evidence_gate",
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
            "end": "add_ai_message",
            "retry": "retry_check",
        },
    )

    workflow.add_edge(
        "add_ai_message",
        END,
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
        "retry_failed",
        END,
    )

    checkpointer = MemorySaver()

    return_flow = workflow.compile(checkpointer=checkpointer)

    # return_flow = workflow.compile()

    # generate and save the graph visualization
    graph_image = return_flow.get_graph().draw_mermaid_png()
    with open("test_dig.png", "wb") as f:
        f.write(graph_image)

    return return_flow


graph = build_graph()

import uuid 

def run_search_agent(query: str, session_id: str):

    print("============ API -> AGENT ==============")

    initial_state = {
        "original_query": query,
        "working_query": query,
        "retry": {
            "count": 0,
            "max_retries": 3,
        },
    }

    config = {
        "run_name": "credit_card_spend",
        "tags": [
            "chatbot",
            "user-query",
        ],
        "metadata": {
            "session_id": session_id,
            "interface": "streamlit",
        },
        "configurable": {
            "thread_id": session_id,
        },
    }

    final_state = graph.invoke(
        initial_state,
        config=config,
    )
    print("========== GRAPH DEBUG ==========")
    print("QUERY:", query)
    print("FINAL ORIGINAL QUERY:", final_state.get("original_query"))
    print("FINAL WORKING QUERY:", final_state.get("working_query"))
    print("AGENT DECISION:", final_state.get("agent_decision"))
    print("RETRIEVAL RESULT:", final_state.get("retrieval_result"))
    print("RERANK RESULT:", final_state.get("rerank_result"))
    print("SQL GENERATION:", final_state.get("sql_generation"))
    print("SQL EXECUTION:", final_state.get("sql_execution"))
    print("ANSWER DRAFT:", final_state.get("answer_draft"))
    print("FINAL EVALUATION:", final_state.get("final_evaluation"))
    print("RETRY:", final_state.get("retry"))
    print("RETRY FEEDBACK:", final_state.get("retry_feedback"))
    print("FINAL RESPONSE:", final_state.get("final_response"))
    print("=================================")    

    return final_state


def stream_search_agent(query: str, session_id: str) -> Generator:

    print("============ API -> AGENT ==============")

    initial_state = {
        "original_query": query,
        "working_query": query,
        "retry": {
            "count": 0,
            "max_retries": 3,
        },
    }

    config = {"configurable": {"thread_id": session_id}}

    for event in graph.stream(
        initial_state,
        config=config,
        stream_mode="updates",
    ):
        yield event
