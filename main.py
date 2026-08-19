from fastapi import FastAPI
from src.api.v1.routes.upload_route import upload_router
from src.api.v1.routes import graph_route, query_route

app = FastAPI()

app.include_router(upload_router)
app.include_router(query_route.router)
app.include_router(graph_route.router)

# To run
# uv run uvicorn main:app --reload
# streamlit run src/ui/streamlit.py