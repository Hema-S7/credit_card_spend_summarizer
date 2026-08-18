from fastapi import FastAPI

from src.api.v1.routes.query import (
    router as query_router,
)

app = FastAPI(
    title="Credit Card Spend Summarizer",
    version="1.0.0",
)


app.include_router(
    query_router,
    prefix="/api/v1",
)


@app.get("/")
def health_check():

    return {
        "status": "running",
        "service": "credit-card-spend-summarizer",
    }
