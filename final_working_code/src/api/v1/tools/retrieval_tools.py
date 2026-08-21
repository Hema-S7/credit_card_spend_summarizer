# from typing import Any

# from psycopg.rows import dict_row

# from src.core.config import RRF_K
# from src.core.db import (
#     get_raw_connection,
#     get_vector_store,
# )

# # ─────────────────────────────────────────────
# # Full Text Search
# # ─────────────────────────────────────────────


# def search_fts(
#     query: str,
#     k: int,
#     collection_name: str,
# ) -> list[dict[str, Any]]:
#     """
#     Search knowledge base using PostgreSQL full-text search.
#     """

#     sql = """
#         SELECT
#             e.document AS content,
#             e.cmetadata AS metadata,

#             ts_rank(
#                 to_tsvector('english', e.document),
#                 plainto_tsquery('english', %(query)s)
#             ) AS fts_rank

#         FROM langchain_pg_embedding e

#         JOIN langchain_pg_collection c
#             ON c.uuid = e.collection_id

#         WHERE c.name = %(collection)s

#         AND to_tsvector('english', e.document)
#             @@ plainto_tsquery('english', %(query)s)

#         ORDER BY fts_rank DESC

#         LIMIT %(k)s;
#     """

#     with get_raw_connection() as conn:

#         with conn.cursor(row_factory=dict_row) as cur:

#             cur.execute(
#                 sql,
#                 {
#                     "query": query,
#                     "collection": collection_name,
#                     "k": k,
#                 },
#             )

#             rows = cur.fetchall()

#     return [
#         {
#             "content": row["content"],
#             "metadata": row["metadata"],
#             "fts_rank": round(
#                 float(row["fts_rank"]),
#                 4,
#             ),
#         }
#         for row in rows
#     ]


# # ─────────────────────────────────────────────
# # Vector Search
# # ─────────────────────────────────────────────


# def search_vector(
#     query: str,
#     k: int,
#     collection_name: str,
# ) -> list[dict[str, Any]]:
#     """
#     Search knowledge base using semantic similarity.
#     """

#     vector_store = get_vector_store(collection_name)

#     docs = vector_store.similarity_search(
#         query,
#         k=k,
#     )

#     return [
#         {
#             "content": doc.page_content,
#             "metadata": doc.metadata,
#         }
#         for doc in docs
#     ]


# # ─────────────────────────────────────────────
# # Hybrid Search using RRF
# # ─────────────────────────────────────────────


# def search_hybrid(
#     query: str,
#     k: int,
#     collection_name: str,
# ) -> list[dict[str, Any]]:
#     """
#     Hybrid search:
#         Vector + FTS + RRF
#     """

#     vector_results = search_vector(
#         query,
#         k,
#         collection_name,
#     )

#     fts_results = search_fts(
#         query,
#         k,
#         collection_name,
#     )

#     rrf_scores: dict[str, float] = {}

#     chunk_map: dict[str, dict[str, Any]] = {}

#     # Vector ranking

#     for rank, doc in enumerate(
#         vector_results,
#         start=1,
#     ):

#         key = doc["content"][:120]

#         rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (RRF_K + rank))

#         chunk_map[key] = {
#             "content": doc["content"],
#             "metadata": doc["metadata"],
#         }

#     # FTS ranking

#     for rank, doc in enumerate(
#         fts_results,
#         start=1,
#     ):

#         key = doc["content"][:120]

#         rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (RRF_K + rank))

#         chunk_map[key] = {
#             "content": doc["content"],
#             "metadata": doc["metadata"],
#         }

#     ranked = sorted(
#         rrf_scores.items(),
#         key=lambda x: x[1],
#         reverse=True,
#     )

#     results = []

#     for key, score in ranked[:k]:

#         results.append(
#             {
#                 **chunk_map[key],
#                 "rrf_score": round(
#                     score,
#                     6,
#                 ),
#             }
#         )

#     return results


from typing import Any

from psycopg.rows import dict_row

from src.core.config import RRF_K
from src.core.db import (
    get_raw_connection,
    get_embeddings,
)


def _build_metadata(
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine stored JSON metadata with useful
    columns from multimodal_chunks.
    """

    metadata = row.get("metadata") or {}

    metadata = dict(metadata)

    metadata["page"] = row.get("page_number")
    metadata["section"] = row.get("section")
    metadata["source"] = row.get("source_file")
    metadata["chunk_id"] = str(row.get("id"))

    return metadata


# ============================================================
# Full Text Search
# ============================================================


def search_fts(
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    """
    PostgreSQL full-text search over multimodal_chunks.
    """
    print("in fts search")
    sql = """
        SELECT
            id,
            content,
            page_number,
            section,
            source_file,
            metadata,

            ts_rank(
                to_tsvector(
                    'english',
                    COALESCE(content, '')
                ),
                plainto_tsquery(
                    'english',
                    %(query)s
                )
            ) AS fts_rank

        FROM multimodal_chunks

        WHERE
            to_tsvector(
                'english',
                COALESCE(content, '')
            )
            @@ plainto_tsquery(
                'english',
                %(query)s
            )

        ORDER BY fts_rank DESC

        LIMIT %(k)s;
    """

    with get_raw_connection() as conn:

        with conn.cursor(row_factory=dict_row) as cur:

            cur.execute(
                sql,
                {
                    "query": query,
                    "k": k,
                },
            )

            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "metadata": _build_metadata(row),
            "fts_rank": float(row["fts_rank"]),
        }
        for row in rows
    ]


# ============================================================
# Semantic / Vector Search
# ============================================================


def search_vector(
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    """
    Semantic search over multimodal_chunks.

    Uses cosine distance via pgvector's <=> operator.

    vector_score is converted to cosine similarity:

        similarity = 1 - cosine_distance

    Higher score = more similar.
    """
    print("in vector search")

    embeddings = get_embeddings()

    query_embedding = embeddings.embed_query(query)

    # Convert embedding into pgvector-compatible literal.
    vector_literal = "[" + ",".join(str(value) for value in query_embedding) + "]"

    sql = """
        SELECT
            id,
            content,
            page_number,
            section,
            source_file,
            metadata,

            1 - (
                embedding
                <=> %(embedding)s::vector
            ) AS vector_score

        FROM multimodal_chunks

        WHERE embedding IS NOT NULL

        ORDER BY
            embedding
            <=> %(embedding)s::vector

        LIMIT %(k)s;
    """

    with get_raw_connection() as conn:

        with conn.cursor(row_factory=dict_row) as cur:

            cur.execute(
                sql,
                {
                    "embedding": vector_literal,
                    "k": k,
                },
            )

            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "metadata": _build_metadata(row),
            "vector_score": float(row["vector_score"]),
        }
        for row in rows
    ]


# ============================================================
# Hybrid Search
# ============================================================


def search_hybrid(
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval:

        FTS
         +
        Vector
         ↓
        RRF
         ↓
        Top K
    """
    print("in hybrid search")
    
    vector_results = search_vector(
        query=query,
        k=k,
    )

    fts_results = search_fts(
        query=query,
        k=k,
    )

    rrf_scores: dict[str, float] = {}

    chunk_map: dict[
        str,
        dict[str, Any],
    ] = {}

    # --------------------------------------------------------
    # Vector results
    # --------------------------------------------------------

    for rank, doc in enumerate(
        vector_results,
        start=1,
    ):

        # Keeping our previously agreed
        # content[:120] deduplication key.
        key = doc["content"][:120]

        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (RRF_K + rank)

        if key not in chunk_map:

            chunk_map[key] = {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "fts_rank": None,
                "vector_score": None,
            }

        chunk_map[key]["vector_score"] = doc.get("vector_score")

    # --------------------------------------------------------
    # FTS results
    # --------------------------------------------------------

    for rank, doc in enumerate(
        fts_results,
        start=1,
    ):

        key = doc["content"][:120]

        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (RRF_K + rank)

        if key not in chunk_map:

            chunk_map[key] = {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "fts_rank": None,
                "vector_score": None,
            }

        chunk_map[key]["fts_rank"] = doc.get("fts_rank")

    # --------------------------------------------------------
    # Sort by RRF
    # --------------------------------------------------------

    ranked = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []

    for rank, (key, rrf_score) in enumerate(
        ranked[:k],
        start=1,
    ):

        results.append(
            {
                **chunk_map[key],
                "rrf_score": rrf_score,
                "rank": rank,
            }
        )

    return results
