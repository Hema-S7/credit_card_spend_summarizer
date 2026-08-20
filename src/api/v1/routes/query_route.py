from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import run_graph_query, stream_graph_query
from src.core.guardrails import GuardrailViolation

router = APIRouter(prefix="/api/v1/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
def graph_query_endpoint(request: QueryRequest) -> QueryResponse:
    try:
        return run_graph_query(request.query, request.session_id)
    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        ) from violation


@router.post("/stream")
def graph_query_stream_endpoint(request: QueryRequest):

    try:

        return StreamingResponse(
            stream_graph_query(
                request.query,
                request.session_id,
            ),
            media_type="text/event-stream",
        )

    except GuardrailViolation as violation:

        raise HTTPException(
            status_code=400,
            detail={
                "guardrail": violation.guard,
                "message": violation.message,
            },
        ) from violation

