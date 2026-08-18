from typing import Any

import cohere

from src.core.config import (
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    RERANK_TOP_N,
)


def get_cohere_client():

    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY is not configured.")

    return cohere.ClientV2(api_key=COHERE_API_KEY)


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    top_n: int | None = None,
) -> dict[str, Any]:
    """
    Rerank retrieved documents using Cohere.

    Input:
        Top-K documents from FTS/vector/hybrid

    Output:
        Top-N reranked documents
    """

    n = top_n or RERANK_TOP_N

    if not documents:

        return {
            "status": "no_results",
            "model": COHERE_RERANK_MODEL,
            "candidates_received": 0,
            "top_n": n,
            "documents": [],
        }

    try:

        co = get_cohere_client()

        rerank_response = co.rerank(
            model=COHERE_RERANK_MODEL,
            query=query,
            documents=[doc["content"] for doc in documents],
            top_n=n,
        )

        reranked_documents = []

        for result in rerank_response.results:

            original_doc = documents[result.index]

            reranked_documents.append(
                {
                    "content": original_doc["content"],
                    "metadata": original_doc["metadata"],
                    # Preserve retrieval scores
                    "fts_rank": original_doc.get("fts_rank"),
                    "vector_score": original_doc.get("vector_score"),
                    "rrf_score": original_doc.get("rrf_score"),
                    # Add Cohere score
                    "rerank_score": result.relevance_score,
                    "rank": result.index + 1,
                }
            )

        return {
            "status": "success",
            "model": COHERE_RERANK_MODEL,
            "candidates_received": len(documents),
            "top_n": n,
            "documents": reranked_documents,
        }

    except Exception:

        return {
            "status": "error",
            "model": COHERE_RERANK_MODEL,
            "candidates_received": len(documents),
            "top_n": n,
            "documents": [],
            "error": ("Document reranking failed."),
        }
