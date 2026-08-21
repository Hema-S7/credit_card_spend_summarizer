from src.api.v1.states.rag_state import RAGState
from src.core.db import get_embeddings

from langchain_core.documents import Document

import os
import psycopg

from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from dotenv import load_dotenv

load_dotenv()

_raw_conn = os.getenv("PG_CONNECTION_STRING_FTS")


# ============================================================
# VECTOR SEARCH
# ============================================================


def vector_search_node(
    state: RAGState, source_file: str = "KB_Credit_Card_Spend_Summarizer.pdf"
):

    print("====== INSIDE vector_search_node")

    query_embedding = get_embeddings().embed_query(state["vector_query"])

    sql = """
        SELECT
            id,
            doc_id,
            content,
            chunk_type,
            element_type,
            image_path,
            mime_type,
            page_number,
            section,
            source_file,
            position,

            1 - (embedding <=> %(embedding)s) AS similarity

        FROM multimodal_chunks

        WHERE source_file = %(source_file)s

        ORDER BY embedding <=> %(embedding)s

        LIMIT %(k)s;
    """

    with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:

        register_vector(conn)

        with conn.cursor() as cur:

            cur.execute(
                sql,
                {
                    "embedding": query_embedding,
                    "source_file": source_file,
                    "k": 20,
                },
            )

            rows = cur.fetchall()

    docs = [
        Document(
            page_content=row["content"],
            metadata={
                "id": row["id"],
                "doc_id": str(row["doc_id"]),
                "chunk_type": row["chunk_type"],
                "element_type": row["element_type"],
                "image_path": row["image_path"],
                "mime_type": row["mime_type"],
                "page_number": row["page_number"],
                "section": row["section"],
                "source_file": row["source_file"],
                "position": row["position"],
                "similarity": round(float(row["similarity"]), 4),
            },
        )
        for row in rows
    ]

    print("Vector Retrieved:", len(docs))

    return {"retrieved_docs": docs}


# ============================================================
# FULL TEXT SEARCH
# ============================================================


def fts_search_node(
    state: RAGState, source_file: str = "KB_Credit_Card_Spend_Summarizer.pdf"
):

    print("====== INSIDE fts_search_node")

    sql = """

        SELECT
            id,
            doc_id,
            content,
            chunk_type,
            element_type,
            image_path,
            mime_type,
            page_number,
            section,
            source_file,
            position,

            ts_rank(
                to_tsvector(
                    'english',
                    content
                ),
                plainto_tsquery(
                    'english',
                    %(query)s
                )
            ) AS fts_rank


        FROM multimodal_chunks


        WHERE source_file = %(source_file)s

        AND to_tsvector(
                'english',
                content
            )
            @@ plainto_tsquery(
                'english',
                %(query)s
            )


        ORDER BY fts_rank DESC


        LIMIT %(k)s;

    """

    with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:

        with conn.cursor() as cur:

            cur.execute(
                sql,
                {
                    "query": state["vector_query"],
                    "source_file": source_file,
                    "k": 20,
                },
            )

            rows = cur.fetchall()

    docs = [
        Document(
            page_content=row["content"],
            metadata={
                "id": row["id"],
                "doc_id": str(row["doc_id"]),
                "chunk_type": row["chunk_type"],
                "element_type": row["element_type"],
                "image_path": row["image_path"],
                "mime_type": row["mime_type"],
                "page_number": row["page_number"],
                "section": row["section"],
                "source_file": row["source_file"],
                "position": row["position"],
                "fts_rank": round(float(row["fts_rank"]), 4),
            },
        )
        for row in rows
    ]

    print("FTS Retrieved:", len(docs))

    return {"retrieved_docs": docs}


# ============================================================
# HYBRID SEARCH (VECTOR + FTS + RRF)
# ============================================================


def hybrid_search_node(
    state: RAGState, source_file: str = "KB_Credit_Card_Spend_Summarizer.pdf"
):

    print("====== INSIDE hybrid_search_node")

    query_embedding = get_embeddings().embed_query(state["vector_query"])

    vector_sql = """

        SELECT

            id,
            doc_id,
            content,
            chunk_type,
            element_type,
            image_path,
            mime_type,
            page_number,
            section,
            source_file,
            position,


            1 - (
                embedding <=> %(embedding)s
            ) AS similarity


        FROM multimodal_chunks


        WHERE source_file = %(source_file)s


        ORDER BY embedding <=> %(embedding)s


        LIMIT 20;

    """

    fts_sql = """

        SELECT

            id,
            doc_id,
            content,
            chunk_type,
            element_type,
            image_path,
            mime_type,
            page_number,
            section,
            source_file,
            position,


            ts_rank(
                to_tsvector(
                    'english',
                    content
                ),

                plainto_tsquery(
                    'english',
                    %(query)s
                )

            ) AS fts_rank



        FROM multimodal_chunks



        WHERE source_file = %(source_file)s


        AND to_tsvector(
                'english',
                content
            )

            @@ plainto_tsquery(
                'english',
                %(query)s
            )



        ORDER BY fts_rank DESC



        LIMIT 20;

    """

    with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:

        register_vector(conn)

        with conn.cursor() as cur:

            cur.execute(
                vector_sql, {"embedding": query_embedding, "source_file": source_file}
            )

            vector_rows = cur.fetchall()

            cur.execute(
                fts_sql, {"query": state["vector_query"], "source_file": source_file}
            )

            fts_rows = cur.fetchall()

    vector_docs = [
        Document(
            page_content=row["content"],
            metadata={
                "id": row["id"],
                "doc_id": str(row["doc_id"]),
                "chunk_type": row["chunk_type"],
                "element_type": row["element_type"],
                "image_path": row["image_path"],
                "mime_type": row["mime_type"],
                "page_number": row["page_number"],
                "section": row["section"],
                "source_file": row["source_file"],
                "position": row["position"],
                "similarity": float(row["similarity"]),
            },
        )
        for row in vector_rows
    ]

    fts_docs = [
        Document(
            page_content=row["content"],
            metadata={
                "id": row["id"],
                "doc_id": str(row["doc_id"]),
                "chunk_type": row["chunk_type"],
                "element_type": row["element_type"],
                "image_path": row["image_path"],
                "mime_type": row["mime_type"],
                "page_number": row["page_number"],
                "section": row["section"],
                "source_file": row["source_file"],
                "position": row["position"],
                "fts_rank": float(row["fts_rank"]),
            },
        )
        for row in fts_rows
    ]

    print("Vector:", len(vector_docs), "FTS:", len(fts_docs))

    # -------------------------
    # Reciprocal Rank Fusion
    # -------------------------

    scores = {}

    docs = {}

    for rank, doc in enumerate(vector_docs):

        key = doc.metadata["id"]

        scores[key] = scores.get(key, 0) + (1 / (60 + rank + 1))

        docs[key] = doc

    for rank, doc in enumerate(fts_docs):

        key = doc.metadata["id"]

        scores[key] = scores.get(key, 0) + (1 / (60 + rank + 1))

        docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)

    hybrid_docs = [docs[key] for key in ranked[:20]]

    print("Hybrid Retrieved:", len(hybrid_docs))

    return {"retrieved_docs": hybrid_docs}
