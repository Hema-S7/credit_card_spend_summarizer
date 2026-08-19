# nodes we want
# 1. vector_search (top-k=20)
# 2. rerank
# 3. generate_answer

import json
import os
import cohere
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal
from src.api.v1.states.rag_state import RAGState
from src.api.v1.tools.vector_search_tool import (
    vector_search_node,
    fts_search_node,
    hybrid_search_node,
)
from src.api.v1.schemas.query_schema import AIResponse
from src.core.db import get_sql_database

load_dotenv()


def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    )


class RouteDecision(BaseModel):
    route: Literal["VECTOR_DB", "RDBMS", "HYBRID_ROUTE"]
    reason: str  # for debugging
    vector_query: str
    rdbms_query: str

class SearchDecision(BaseModel):
    search_strategy: Literal["VECTOR", "FTS", "HYBRID"]
    reason: str


class EvaluationResult(BaseModel):
    is_correct: bool
    score: int
    reason: str
    missing_information: str
    should_retry: bool


def router_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                       You are a query router for an Agentic RAG System.
                       Classify the query and rewrite it for the correct systems.  
                       
                       'VECTOR_DB' -  the auery asks about policies, procedures, guides, guidelines, 
                       regulations, or any topic that requires reading text documents

                       'RBDMS - the query asks about products, product prices, stock/inventory, 
                       product categories, customer orders, order items, or anything answerable 
                       from a structrured e-commerce database tables: 
                       products, categories, orders, order_items

                       HYBRID:
                        - questions requiring BOTH document knowledge and database information.

                        Examples:

                        Question:
                        "What is ESS policy?"
                        Route:
                        VECTOR_DB

                        Question:
                        "Show products in category electronics"
                        Route:
                        RDBMS

                        Question:
                        "Explain ESS policy and show products affected by it"
                        Route:
                        HYBRID

                        After deciding route, rewrite the query.

                        For VECTOR_DB:
                        - Fill vector_query
                        - Leave rdbms_query empty


                        For RDBMS:
                        - Fill rdbms_query
                        - Leave vector_query empty


                        For HYBRID:
                        Split the question:

                        vector_query:
                    document/policy part

                    rdbms_query:
                    product/database part


                    Example:

                    Input:
                    "Explain ESS policy and show product detail 1"


                    Output:

                    route:
                    HYBRID_ROUTE


                    vector_query:
                    "Explain ESS policy"


                    rdbms_query:
                    "Show product detail 1"

                      Reply with the route and one sentence of reason.
                   """,
            ),
            (
                "human",
                """
                   Question:
                   {query}
                """,
            ),
        ]
    )

    chain = prompt | structured_llm
    decision = chain.invoke({"query": state["query"]})
    print(f"[router_node's decision]: {decision.route} and reason: {decision.reason}")
    print(decision.vector_query, decision.rdbms_query)
    return {
        **state,
        "route": decision.route,
        "vector_query": decision.vector_query,
        "rdbms_query": decision.rdbms_query,
    }


def vector_router_node(state: RAGState) -> RAGState:

    llm = _get_llm()

    structured_llm = llm.with_structured_output(SearchDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a search strategy router.

                Decide the best retrieval strategy:

                VECTOR:
                - conceptual questions
                - policies
                - procedures
                - explanations
                - documents where meaning matters

                FTS:
                - exact terms
                - names
                - codes
                - specific phrases

                HYBRID:
                - when both semantic meaning and exact keywords matter

                Return only one strategy.
                """,
            ),
            (
                "human",
                """
                Query:
                {query}
                """,
            ),
        ]
    )

    chain = prompt | structured_llm

    decision = chain.invoke({"query": state["vector_query"]})

    print(
        f"[vector_router's decision] {decision.search_strategy} and reason: {decision.reason}"
    )

    return {**state, "search_strategy": decision.search_strategy}


def nl2sql_node(state: RAGState) -> RAGState:
    print("About to generate nl2sql")
    # connect to LLM
    llm = _get_llm()
    # connect to rdbms
    db = get_sql_database()
    # get the tables' live schema
    schema_info = db.get_table_info()
    # write the system prompt and pass on the schema to get only sql query
    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                   You are a PostgreSQL expert. Given the database schema below,
                   write a single valid SELECT query that answers the user's question.


                   Rules:
                   - Return ONLY the raw SQL — no explanation, no summary, no markdown fences, no backticks.
                   - Use only the tables and columns present in the schema.
                   - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
                   - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
                   - For product or text searches: NEVER search for the full multi-word phrase as one
                       ILIKE pattern. Instead, split the search into individual meaningful keywords
                       and OR them together across both name and description columns.
                       Example — user asks "wireless headset":
                           WHERE (name ILIKE '%wireless%' OR description ILIKE '%wireless%')
                           OR (name ILIKE '%headset%'  OR description ILIKE '%headset%')
                           OR (name ILIKE '%headphones%' OR description ILIKE '%headphones%')
                       Use your knowledge of synonyms (headset/headphones, laptop/notebook, etc.)
                       to cast a wider net when the exact term may not match.
                  
                   Database schema:
                   {schema}
               """,
            ),
            (
                "human",
                """
                   Question:
                   {question}
               """,
            ),
        ]
    )
    # preprare the chain and invoke with a query
    sql_chain = sql_prompt | llm
    # look for sql query only
    raw_sql = sql_chain.invoke({"schema": schema_info, "question": state["rdbms_query"]})
    print("========GENERATED raw_sql query is: =====")
    print(raw_sql.content)
    generated_sql = raw_sql.content

    # execute the generated sql query  to get the outout from RDMBS
    try:
        sql_result = db.run(generated_sql)
    except Exception as err:
        sql_result = f"Generated SQL execution error: {err}"

    # connect to LLM to get the natural language response
    structured_llm = llm.with_structured_output(AIResponse)
    nl_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful data analyst. Answer the user's question using
               the SQL query results below. Be concise and format numbers/lists clearly.
               Set policy_citations to empty string,
               page_no to 'N/A', and document_name to 'agentic_rag_db'.
               - Do NOT execute INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements
               even if requested.
               - Politely deny when users are asking for these actions in their queries.
               - Never use tech jargons in your response""",
            ),
            (
                "human",
                "Question: {query}\n\n"
                "SQL Used:\n{sql}\n\n"
                "Query Results:\n{result}",
            ),
        ]
    )

    nl_chain = nl_answer_prompt | structured_llm
    answer = nl_chain.invoke(
        {"query": state["rdbms_query"], "sql": generated_sql, "result": sql_result}
    )
    print("[nl2sql_node] Answer generated.")
    response = answer.model_dump()
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = generated_sql
    # return the sql query is RAGState
    # and also the output in sql_result of RAGState
    return {
        #**state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        "sql_response": response,
    }


def rerank_node(state: RAGState):
    # establish connection with the cohere reranking model
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    # send the query and the retrieved_docs to the reranking model

    docs = state["retrieved_docs"]

    print("=======3. INSIDE rerank_node. Before calling reranker =========")
    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=state["vector_query"],
        documents=[doc.page_content for doc in docs],
        top_n=5,
    )

    # Map Cohere result indices back to LangChain Document objects
    reranked_docs = [docs[r.index] for r in rerank_response.results]

    print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")
    for i, r in enumerate(rerank_response.results):
        print(
            f"  Rank {i+1} | Cohere score: {r.relevance_score:.4f} | original index: {r.index}"
        )

    return {"reranked_docs": reranked_docs}  # **state,


def generate_answer_node(state: RAGState):

    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)

    print("=========4. INSIDE GENERATE ANSWER NODE==========")

    # ==================================================
    # DOCUMENT RESULTS (ANSWER + METADATA)
    # ==================================================

    document_context = []
    document_metadata = []

    for doc in state.get("reranked_docs", []):

        document_context.append(doc.page_content)

        document_metadata.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "metadata": doc.metadata,
            }
        )

    documents_text = (
        "\n\n".join(document_context) if document_context else "Not available"
    )

    # ==================================================
    # SQL RESULTS (ANSWER + METADATA)
    # ==================================================

    sql_answer = "Not available"
    sql_query = None
    sql_metadata = {}

    if state.get("sql_response"):

        sql_response = state["sql_response"]

        # preserve SQL generated answer
        sql_answer = sql_response.get("response", "No SQL answer")

        # preserve SQL query
        sql_query = sql_response.get("sql_query_executed")

        # preserve complete SQL metadata
        sql_metadata = {
            "document_name": sql_response.get("document_name"),
            "page_no": sql_response.get("page_no"),
            "policy_citations": sql_response.get("policy_citations"),
        }

    elif state.get("sql_result"):

        sql_answer = state["sql_result"]

        sql_query = state.get("generated_sql")

    print("SQL ANSWER:")
    print(sql_answer)

    print("SQL QUERY:")
    print(sql_query)

    # ==================================================
    # FINAL GENERATION
    # ==================================================

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant.

                Answer the user question using both sources.

                Rules:

                1. Document information:
                   - use for policies, procedures,
                     guidelines and explanations.

                2. SQL information:
                   - use for products, prices,
                     inventory and orders.

                3. If both exist, combine both answers.

                4. Preserve citations.

                Output fields:

                document_name:
                    document used for document answers

                page_no:
                    document pages used

                policy_citations:
                    important document references

                sql_query_executed:
                    SQL query used for database answer

                If no document information exists:
                    document_name = "agentic_rag_db"
                    page_no = "N/A"
                    policy_citations = "N/A"

                Do not mention internal systems.
                """,
            ),
            (
                "human",
                """
                User Question:

                {query}


                Document Answer:

                {documents}


                Document Metadata:

                {document_metadata}


                Database Answer:

                {sql_answer}


                SQL Query:

                {sql_query}


                Database Metadata:

                {sql_metadata}

                """,
            ),
        ]
    )

    chain = prompt | structured_llm

    result = chain.invoke(
        {
            "query": state["query"],
            "documents": documents_text,
            "document_metadata": json.dumps(document_metadata, indent=2),
            "sql_answer": sql_answer,
            "sql_query": sql_query or "N/A",
            "sql_metadata": json.dumps(sql_metadata, indent=2),
        }
    )

    final_response = result.model_dump()

    # Preserve raw data also for UI/debugging
    final_response["document_chunks"] = document_metadata

    final_response["sql_metadata"] = sql_metadata

    return {**state, "response": final_response}


def evaluate_answer_node(state: RAGState):

    print("=========5. INSIDE EVALUATION NODE==========")

    llm = _get_llm()

    evaluator_llm = llm.with_structured_output(EvaluationResult)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an answer quality evaluator.

                Your job:
                Compare the user's question with the final answer.

                Check:

                1. Did the answer address the complete user query?
                2. If the query required multiple parts,
                   are all parts answered?
                3. Are citations present when documents were used?
                4. Is the answer based only on provided context?
                5. Is anything important missing?

                Score:
                0-10

                Return:
                - is_correct
                - score
                - reason
                - missing_information

                Check if the answer completely satisfies the user.

                Retry rules:

                score >= 5:
                    should_retry=false

                score < 5:
                    should_retry=true

                Do not rewrite the answer.
                """,
            ),
            (
                "human",
                """
                User Question:

                {query}


                Final Answer:

                {answer}


                Available Document Context:

                {documents}


                Available SQL Information:

                {sql}

                """,
            ),
        ]
    )

    chain = prompt | evaluator_llm

    result = chain.invoke(
        {
            "query": state["query"],
            "answer": state["response"]["response"],
            "documents": json.dumps(state.get("document_chunks", [])),
            "sql": state.get("generated_sql", "N/A"),
        }
    )

    evaluation = result.model_dump()

    print("Evaluation:")
    print(evaluation)

    return {**state, "evaluation": evaluation}


def hybrid_route_node(state: RAGState):
    print("====== HYBRID ROUTE NODE ======")
    return state


def evaluation_router(state: RAGState):

    evaluation = state["evaluation"]

    retry_count = state.get("retry_count", 0)

    print("========= EVALUATION ROUTER =========")
    print("Score:", evaluation["score"])
    print("Retry count:", retry_count)

    # answer accepted
    if evaluation["should_retry"] is False:
        return "END"

    # maximum retry reached
    if retry_count >= 3:
        return "END"

    return "RETRY"


def rewrite_query_node(state: RAGState):

    print("====== QUERY REWRITE NODE ======")

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a query improvement agent.

                Improve the user's query so retrieval gives a better answer.

                Use:
                - missing information
                - previous failed answer
                - original intent

                Add important keywords.
                Make ambiguous requirements clear.

                Return only the rewritten query.
                """,
            ),
            (
                "human",
                """
                Original Question:

                {query}


                Previous Answer:

                {answer}


                Missing Information:

                {missing}

                """,
            ),
        ]
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "query": state["original_query"],
            "answer": state["response"].get("response", ""),
            "missing": state["evaluation"].get("missing_information", ""),
        }
    )

    new_query = result.content.strip()

    print("OLD QUERY:")
    print(state["query"])

    print("NEW QUERY:")
    print(new_query)

    return {
        "query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "route": None,
        "vector_query": "",
        "rdbms_query": "",
        "search_strategy": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "evaluation": {},
        "generated_sql": None,
        "sql_result": None,
        "sql_response": None,
    }

def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("router", router_node)
    workflow.add_node("nl2sql", nl2sql_node)
    workflow.add_node("hybrid_router", hybrid_route_node)
    workflow.add_node("vector_router", vector_router_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("fts_search", fts_search_node)
    workflow.add_node("hybrid_search", hybrid_search_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate_answer", evaluate_answer_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    # the following is the starting point
    workflow.set_entry_point("router")

    # conditional routing: "vectordb" -> vector_search (or) "rdbms" -> nl2sql
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "VECTOR_DB": "vector_router",
            "RDBMS": "nl2sql",
            "HYBRID_ROUTE": "hybrid_router",
        },
    )

    workflow.add_conditional_edges(
        "vector_router",
        lambda state: state["search_strategy"],
        {"VECTOR": "vector_search", "FTS": "fts_search", "HYBRID": "hybrid_search"},
    )
    workflow.add_edge("hybrid_router", "vector_router")

    workflow.add_edge("hybrid_router", "nl2sql")

    workflow.add_edge("vector_search", "rerank")
    workflow.add_edge("fts_search", "rerank")
    workflow.add_edge("hybrid_search", "rerank")
    workflow.add_edge(["rerank", "nl2sql"], "generate_answer")
    workflow.add_edge("generate_answer","evaluate_answer")

    workflow.add_conditional_edges(
        "evaluate_answer", evaluation_router, {"END": END, "RETRY": "rewrite_query"}
    )
    workflow.add_edge("rewrite_query", "router")

    search_agent = workflow.compile()

    # generating and saving the graph visualization
    graph_image = search_agent.get_graph().draw_mermaid_png()
    with open("search_agent.png", "wb") as f:
        f.write(graph_image)

    return search_agent


rag_graph = build_rag_graph()


# def run_search_agent(query: str):
#     print("============1. INSIDE run_search_agent ")
#     initial_state = {
#        "query": query,
#        "retrieved_docs": [],
#        "reranked_docs": [],
#        "response": {},
#    }

#     final_state = rag_graph.invoke(initial_state)
#     return final_state["response"]


# non streaming response
def run_search_agent(query: str, session_id: str):
    print("============1. INSIDE run_search_agent ")
    initial_state = {
        "query": query,
        "original_query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "evaluation": {},
        "generated_sql": None,
        "sql_result": None,
        "sql_response": None,
        "session_id": session_id,
        "retry_count": 0,
        "route": None,
        "vector_query": "",
        "rdbms_query": "",
        "search_strategy": None,
    }

    final_state = rag_graph.invoke(initial_state)
    print(final_state["response"])
    return final_state["response"]


async def run_search_agent_stream(query: str, session_id: str):
    print("============1. INSIDE run_search_agent ")
    initial_state = {
        "query": query,
        "original_query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "evaluation": {},
        "generated_sql": None,
        "sql_result": None,
        "sql_response": None,
        "session_id": session_id,
        "retry_count": 0,
        "route": None,
        "vector_query": "",
        "rdbms_query": "",
        "search_strategy": None,
    }

    async for event in rag_graph.astream_events(initial_state, version="v1"):
        kind = event["event"]
        # print(kind)

        # if it is a token generated by the chat model
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # format as an Server Side Event data straem payload
                yield f"data: {json.dumps({'token': content})}\n\n"

    yield "data: [DONE]\n\n"
