from fastapi import APIRouter, HTTPException

from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import run_graph_query
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
