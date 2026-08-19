from src.api.v1.agents.graph import run_search_agent
from src.core.guardrails import guard_input, guard_output


def run_graph_query(query: str, session_id: str) -> dict:
    guard_input(query)

    graph_state = run_search_agent(query, session_id)
    result = graph_state.get("answer_draft") or graph_state.get("final_response")

    if not result:
        raise RuntimeError("The search graph did not produce a response.")

    result = {
        **result,
        "response": result.get("response", result.get("answer", "")),
        "policy_citations": result.get(
            "policy_citations", result.get("citations", "N/A")
        ),
    }

    if result["response"]:
        result["response"] = guard_output(result["response"])

    return result
