from src.agents.graph import build_graph

graph = build_graph()

config = {"configurable": {"thread_id": "memory-test-001"}}


# -----------------------------------------
# TURN 1
# -----------------------------------------

query1 = "My name is John. " "What are the 2 top selling credit card variants?"

result1 = graph.invoke(
    {
        "original_query": query1,
        "working_query": query1,
        "retry": {
            "count": 0,
            "max_retries": 3,
        },
    },
    config=config,
)


print("\n==============================")
print("TURN 1")
print("==============================")

print("\nAnswer:")
print(result1.get("answer_draft"))

print("\nConversation History:")
print(result1.get("conversation_history"))


# -----------------------------------------
# TURN 2
# -----------------------------------------

query2 = "What is my name?"

result2 = graph.invoke(
    {
        "original_query": query2,
        "working_query": query2,
        "retry": {
            "count": 0,
            "max_retries": 3,
        },
    },
    config=config,
)


print("\n==============================")
print("TURN 2")
print("==============================")

print("\nAgent Decision:")
print(result2.get("agent_decision"))

print("\nAnswer:")
print(result2.get("answer_draft"))

print("\nConversation History:")
print(result2.get("conversation_history"))

# -----------------------------------------
# TURN 3
# -----------------------------------------

query3 = "give me detaails of the mentioned earlier credit card variants?"

result3 = graph.invoke(
    {
        "original_query": query3,
        "working_query": query3,
        "retry": {
            "count": 0,
            "max_retries": 3,
        },
    },
    config=config,
)


print("\n==============================")
print("TURN 3")
print("==============================")

print("\nWorking query:")
print(result3.get("working_query"))

print("\nAnswer:")
print(result3.get("answer_draft"))

print("\nConversation History:")
print(result3.get("conversation_history"))
