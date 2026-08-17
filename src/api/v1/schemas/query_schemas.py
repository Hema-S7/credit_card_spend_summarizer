from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User's query",
    )

    thread_id: str = Field(
        ...,
        min_length=1,
        description="Conversation thread identifier",
    )


class AIResponse(BaseModel):
    query: str = Field(description="The original query given by the user")

    answer: str = Field(description="The generated response")

    citations: str = Field(
        description="Citations for the documents used to generate the response"
    )

    page_no: str = Field(description="Page number from the retrieved document metadata")

    document_name: str = Field(
        description="Name of the document used to generate the response"
    )

    sql_query_executed: Optional[str] = Field(
        default=None,
        description=(
            "The AI-generated SQL query executed to answer " "the user's query"
        ),
    )
