from typing import Any, Literal, TypedDict

# ─────────────────────────────────────────────
# Conversation
# ─────────────────────────────────────────────


class ConversationMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


# ─────────────────────────────────────────────
# Agent Decision
# ─────────────────────────────────────────────

QueryType = Literal["faq", "analytics", "combined", "conversation", "unrelated"]

RetrievalStrategy = Literal[
    "fts",
    "semantic",
    "hybrid",
]


class AgentDecision(TypedDict):
    query_type: QueryType
    needs_faq: bool
    needs_analytics: bool
    retrieval_strategy: RetrievalStrategy | None


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────


class RetrievedDocument(TypedDict, total=False):
    content: str
    metadata: dict[str, Any]

    rank: int

    fts_rank: float
    vector_score: float
    rrf_score: float


class RetrievalResult(TypedDict):
    strategy: RetrievalStrategy
    candidates: list[RetrievedDocument]
    top_k: int
    status: Literal[
        "success",
        "no_results",
        "error",
    ]


class RerankedDocument(TypedDict):
    content: str
    metadata: dict[str, Any]

    fts_rank: float | None
    vector_score: float | None
    rrf_score: float | None

    rerank_score: float
    rank: int


class RerankResult(TypedDict):
    status: Literal[
        "success",
        "no_results",
        "error",
    ]

    model: str
    candidates_received: int
    top_n: int

    documents: list[RerankedDocument]


class RetrievalEvaluation(TypedDict):
    threshold_passed: bool
    llm_relevance_passed: bool

    threshold_score: float | None
    llm_relevance_score: float | None

    passed: bool
    retry_required: bool

    issues: list[str]


# ─────────────────────────────────────────────
# SQL
# ─────────────────────────────────────────────


class SQLGenerationResult(TypedDict):
    status: Literal[
        "success",
        "error",
    ]

    sql: str | None


class SQLValidationResult(TypedDict):
    passed: bool

    is_read_only: bool
    syntax_valid: bool
    schema_valid: bool
    single_statement: bool

    tables_used: list[str]
    columns_used: list[str]

    issues: list[str]


class SQLExecutionResult(TypedDict):
    status: Literal[
        "success",
        "empty",
        "timeout",
        "error",
    ]

    columns: list[str]
    rows: list[dict[str, Any]]

    row_count: int
    truncated: bool

    error_code: str | None
    error_message: str | None


class SQLEvaluation(TypedDict):
    correct: bool
    complete: bool
    grounded: bool

    score: float

    passed: bool
    retry_required: bool

    issues: list[str]


# ─────────────────────────────────────────────
# Retry
# ─────────────────────────────────────────────

FailureStage = Literal[
    "agent",
    "retrieval",
    "reranking",
    "retrieval_evaluation",
    "nl2sql",
    "sql_validation",
    "sql_execution",
    "sql_evaluation",
    "answer_builder",
    "final_evaluation",
]


class RetryState(TypedDict):
    count: int
    max_retries: int


class RetryFeedback(TypedDict):
    failure_stage: FailureStage

    error_code: str | None

    issues: list[str]


class RetryQuery(TypedDict):
    revised_query: str

    retry_number: int
    failure_stage: str

    issues_addressed: list[str]


# ─────────────────────────────────────────────
# Final Evaluation
# ─────────────────────────────────────────────


class FinalEvaluation(TypedDict):
    correct: bool
    complete: bool
    grounded: bool

    score: float

    passed: bool
    retry_required: bool

    issues: list[str]


# ─────────────────────────────────────────────
# Generic Workflow Error
# ─────────────────────────────────────────────


class WorkflowError(TypedDict):
    stage: str
    error_code: str
    message: str
    retryable: bool


# ─────────────────────────────────────────────
# Main LangGraph State
# ─────────────────────────────────────────────


class AgentState(TypedDict, total=False):

    # Query
    original_query: str
    working_query: str

    # Conversation
    conversation_history: list[ConversationMessage]

    # Agent
    agent_decision: AgentDecision

    # Retrieval branch
    retrieval_result: RetrievalResult
    rerank_result: RerankResult
    retrieval_evaluation: RetrievalEvaluation

    # SQL branch
    sql_generation: SQLGenerationResult
    sql_validation: SQLValidationResult
    sql_execution: SQLExecutionResult
    sql_evaluation: SQLEvaluation

    # Answer
    answer_draft: dict[str, Any]

    # Final evaluation
    final_evaluation: FinalEvaluation

    # Retry
    retry: RetryState
    retry_feedback: RetryFeedback
    retry_query: RetryQuery

    # Error
    workflow_error: WorkflowError

    # Final user response
    final_response: dict[str, Any]
