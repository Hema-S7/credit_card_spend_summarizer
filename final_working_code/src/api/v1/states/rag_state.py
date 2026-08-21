from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
import operator


from typing import Annotated
import operator


class RAGState(TypedDict):

    query: str

    vector_query: str
    rdbms_query: str

    route: str
    search_strategy: str

    retrieved_docs: list[Document]

    reranked_docs: list[Document]

    generated_sql: Annotated[str | None, lambda old, new: new]
    sql_result: Annotated[str | None, lambda old, new: new]
    sql_response: Annotated[dict | None, lambda old, new: new]

    response: dict

    session_id: str

    original_query: str

    retry_count: int

    evaluation: dict
