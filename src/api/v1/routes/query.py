import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.v1.schemas.query_schema import (
    QueryRequest,
    AIResponse,
)

from src.agents.graph import build_graph
from src.core.config import MAX_RETRIES

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/query",
    tags=["Query"],
)


# Build graph once when API starts
graph = build_graph()


def build_initial_state(request: QueryRequest) -> dict:
    """
    Build state for a new graph invocation.

    Do not reset conversation_history here because
    LangGraph checkpointer manages thread state.
    """

    return {
        "original_query": request.query,
        "working_query": request.query,
        "retry": {
            "count": 0,
            "max_retries": MAX_RETRIES,
        },
    }


def get_response_from_result(result: dict) -> AIResponse:
    """
    Return the final validated response.

    retry_exhausted_response writes final_response.
    Normal successful execution currently writes answer_draft.
    """

    response_data = result.get("final_response") or result.get("answer_draft")

    if not response_data:
        raise ValueError("Graph completed without a response.")

    return AIResponse.model_validate(response_data)


# ============================================================
# Normal non-streaming endpoint
# ============================================================


@router.post(
    "",
    response_model=AIResponse,
)
async def process_query(
    request: QueryRequest,
):

    try:

        initial_state = build_initial_state(request)

        config = {"configurable": {"thread_id": request.thread_id}}

        # IMPORTANT:
        # Use ainvoke() when you want the completed graph state.
        result = await graph.ainvoke(
            initial_state,
            config=config,
        )

        return get_response_from_result(result)

    except Exception:

        logger.exception("Failed to process query.")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the query.",
        )


# ============================================================
# Streaming endpoint
# ============================================================


@router.post("/stream")
async def stream_query(
    request: QueryRequest,
):

    initial_state = build_initial_state(request)

    config = {"configurable": {"thread_id": request.thread_id}}

    async def event_generator():

        try:

            # ------------------------------------------------
            # Run COMPLETE graph first.
            #
            # This means:
            # Agent
            # Retrieval / SQL
            # Answer Builder
            # Final Evaluator
            # Retry if necessary
            #
            # Only after validation passes do we stream
            # the user-facing answer.
            # ------------------------------------------------

            result = await graph.ainvoke(
                initial_state,
                config=config,
            )

            response = get_response_from_result(result)

            # ------------------------------------------------
            # Stream validated answer progressively
            # ------------------------------------------------

            words = response.answer.split()

            for word in words:

                event = {
                    "type": "chunk",
                    "content": f"{word} ",
                }

                yield (f"data: " f"{json.dumps(event)}\n\n")

                # Allow event loop to send the chunk
                await asyncio.sleep(0)

            # ------------------------------------------------
            # Send complete structured response at the end
            # ------------------------------------------------

            final_event = {
                "type": "final",
                "data": response.model_dump(),
            }

            yield (f"data: " f"{json.dumps(final_event)}\n\n")

        except Exception:

            logger.exception("Streaming query failed.")

            error_event = {
                "type": "error",
                "message": ("Unable to process the query."),
            }

            yield (f"data: " f"{json.dumps(error_event)}\n\n")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
