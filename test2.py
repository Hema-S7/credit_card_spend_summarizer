from src.core.db import get_raw_connection

sql = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS multimodal_chunks (
    id BIGSERIAL PRIMARY KEY,

    doc_id UUID NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    chunk_type TEXT,
    element_type TEXT,

    content TEXT NOT NULL,

    image_path TEXT,
    mime_type TEXT,

    page_number INTEGER,
    section TEXT,
    source_file TEXT,

    position JSONB,

    embedding VECTOR(1536),

    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


with get_raw_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()


print("Ingestion tables created.")
