from typing import Any

from psycopg.rows import dict_row

from src.core.config import RRF_K
from src.core.db import (
    get_raw_connection,
    get_vector_store,
)

# ─────────────────────────────────────────────
# Full Text Search
# ─────────────────────────────────────────────


def search_fts(
    query: str,
    k: int,
    collection_name: str,
) -> list[dict[str, Any]]:
    """
    Search knowledge base using PostgreSQL full-text search.
    """

    sql = """
        SELECT
            e.document AS content,
            e.cmetadata AS metadata,

            ts_rank(
                to_tsvector('english', e.document),
                plainto_tsquery('english', %(query)s)
            ) AS fts_rank

        FROM langchain_pg_embedding e

        JOIN langchain_pg_collection c
            ON c.uuid = e.collection_id

        WHERE c.name = %(collection)s

        AND to_tsvector('english', e.document)
            @@ plainto_tsquery('english', %(query)s)

        ORDER BY fts_rank DESC

        LIMIT %(k)s;
    """

    with get_raw_connection() as conn:

        with conn.cursor(row_factory=dict_row) as cur:

            cur.execute(
                sql,
                {
                    "query": query,
                    "collection": collection_name,
                    "k": k,
                },
            )

            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "metadata": row["metadata"],
            "fts_rank": round(
                float(row["fts_rank"]),
                4,
            ),
        }
        for row in rows
    ]


# ─────────────────────────────────────────────
# Vector Search
# ─────────────────────────────────────────────


def search_vector(
    query: str,
    k: int,
    collection_name: str,
) -> list[dict[str, Any]]:
    """
    Search knowledge base using semantic similarity.
    """

    vector_store = get_vector_store(collection_name)

    docs = vector_store.similarity_search(
        query,
        k=k,
    )

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


# ─────────────────────────────────────────────
# Hybrid Search using RRF
# ─────────────────────────────────────────────


def search_hybrid(
    query: str,
    k: int,
    collection_name: str,
) -> list[dict[str, Any]]:
    """
    Hybrid search:
        Vector + FTS + RRF
    """

    vector_results = search_vector(
        query,
        k,
        collection_name,
    )

    fts_results = search_fts(
        query,
        k,
        collection_name,
    )

    rrf_scores: dict[str, float] = {}

    chunk_map: dict[str, dict[str, Any]] = {}

    # Vector ranking

    for rank, doc in enumerate(
        vector_results,
        start=1,
    ):

        key = doc["content"][:120]

        rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (RRF_K + rank))

        chunk_map[key] = {
            "content": doc["content"],
            "metadata": doc["metadata"],
        }

    # FTS ranking

    for rank, doc in enumerate(
        fts_results,
        start=1,
    ):

        key = doc["content"][:120]

        rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (RRF_K + rank))

        chunk_map[key] = {
            "content": doc["content"],
            "metadata": doc["metadata"],
        }

    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for key, score in ranked[:k]:

        results.append(
            {
                **chunk_map[key],
                "rrf_score": round(
                    score,
                    6,
                ),
            }
        )

    return results
