from src.api.v1.agents.graph import run_search_agent, stream_search_agent
from src.core.guardrails import guard_input, guard_output
import json


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


def stream_graph_query(query: str, session_id: str):

    guard_input(query)

    for event in stream_search_agent(query, session_id):

        if not event:
            continue

        # Each graph node update
        for node_name, state in event.items():

            if not state:
                continue

            result = state.get("answer_draft") or state.get("final_response")

            if not result:
                continue

            response_text = result.get("response", result.get("answer", ""))

            if response_text:
                response_text = guard_output(response_text)

            payload = {
                **result,
                "response": response_text,
                "policy_citations": result.get(
                    "policy_citations", result.get("citations", "N/A")
                ),
            }

            yield ("data: " + json.dumps(payload) + "\n\n")

    yield "data: [DONE]\n\n"





# async def stream_graph_query(
#     query: str,
#     session_id: str,
# ):

#     guard_input(query)

#     complete_response = ""

#     # Collect full answer
#     async for token in stream_search_agent(
#         query,
#         session_id,
#     ):

#         complete_response += token

#     # OUTPUT GUARDRAIL
#     complete_response = guard_output(complete_response)

#     # Stream the safe response
#     for token in complete_response.split():

#         yield ("data: " + json.dumps({"token": token + " "}) + "\n\n")

#     yield "data: [DONE]\n\n"
