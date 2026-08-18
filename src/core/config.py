import os

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-5.5"


# ─────────────────────────────────────────────
# Cohere
# ─────────────────────────────────────────────

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

COHERE_RERANK_MODEL = "rerank-v3.5"


# ─────────────────────────────────────────────
# PostgreSQL
# ─────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")

PGVECTOR_COLLECTION_NAME = "credit_card_knowledge"


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))

RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

RRF_K = int(os.getenv("RRF_K", "60"))

RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.5"))


# ─────────────────────────────────────────────
# SQL
# ─────────────────────────────────────────────

MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "50"))

QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "10"))


# ─────────────────────────────────────────────
# Retry
# ─────────────────────────────────────────────

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────

API_HOST = os.getenv(
    "API_HOST",
    "0.0.0.0",
)

API_PORT = int(os.getenv("API_PORT", "8000"))


# ─────────────────────────────────────────────
# Text embedding model
# ─────────────────────────────────────────────


OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)
