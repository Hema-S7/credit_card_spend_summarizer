from src.api.v1.agents.graph import build_graph

graph = build_graph()


query = """give card details of james """


initial_state = {
    "original_query": query,
    "working_query": query,
    "conversation_history": [],
    "retry": {
        "count": 0,
        "max_retries": 3,
    },
}


config = {"configurable": {"thread_id": "combined-debug-001"}}


for event in graph.stream(
    initial_state,
    config=config,
    stream_mode="updates",
):

    print("\nEVENT:")
    print(event)
