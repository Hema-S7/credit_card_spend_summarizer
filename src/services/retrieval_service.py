# from typing import Any

# from src.core.config import RETRIEVAL_TOP_K
# from src.core.config import PGVECTOR_COLLECTION_NAME

# from src.tools.retrieval_tools import (
#     search_fts,
#     search_vector,
#     search_hybrid,
# )


# def retrieve_documents(
#     query: str,
#     strategy: str,
#     collection_name: str | None = None,
#     top_k: int | None = None,
# ) -> dict[str, Any]:
#     """
#     Execute retrieval based on Agent decision.

#     Supported strategies:
#         - fts
#         - semantic
#         - hybrid
#     """

#     collection = collection_name or PGVECTOR_COLLECTION_NAME

#     k = top_k or RETRIEVAL_TOP_K

#     try:

#         if strategy == "fts":

#             documents = search_fts(
#                 query=query,
#                 k=k,
#                 collection_name=collection,
#             )

#         elif strategy == "semantic":

#             documents = search_vector(
#                 query=query,
#                 k=k,
#                 collection_name=collection,
#             )

#         elif strategy == "hybrid":

#             documents = search_hybrid(
#                 query=query,
#                 k=k,
#                 collection_name=collection,
#             )

#         else:

#             return {
#                 "strategy": strategy,
#                 "candidates": [],
#                 "top_k": k,
#                 "status": "error",
#                 "error": (f"Unsupported retrieval strategy: {strategy}"),
#             }

#         if not documents:

#             return {
#                 "strategy": strategy,
#                 "candidates": [],
#                 "top_k": k,
#                 "status": "no_results",
#             }

#         return {
#             "strategy": strategy,
#             "candidates": documents,
#             "top_k": k,
#             "status": "success",
#         }

#     except Exception:

#         return {
#             "strategy": strategy,
#             "candidates": [],
#             "top_k": k,
#             "status": "error",
#             "error": ("Retrieval operation failed."),
#         }


from typing import Any

from src.core.config import (
    RETRIEVAL_TOP_K,
)

from src.api.v1.tools.retrieval_tools import (
    search_fts,
    search_vector,
    search_hybrid,
)


def retrieve_documents(
    query: str,
    strategy: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Execute retrieval based on Agent decision.

    Supported:
        fts
        semantic
        hybrid
    """

    k = top_k or RETRIEVAL_TOP_K

    try:

        if strategy == "fts":

            documents = search_fts(
                query=query,
                k=k,
            )

        elif strategy == "semantic":

            documents = search_vector(
                query=query,
                k=k,
            )

        elif strategy == "hybrid":

            documents = search_hybrid(
                query=query,
                k=k,
            )

        else:

            return {
                "strategy": strategy,
                "candidates": [],
                "top_k": k,
                "status": "error",
            }

        if not documents:

            return {
                "strategy": strategy,
                "candidates": [],
                "top_k": k,
                "status": "no_results",
            }

        return {
            "strategy": strategy,
            "candidates": documents,
            "top_k": k,
            "status": "success",
        }

    except Exception as err:

        print(f"Retrieval error: {err}")

        return {
            "strategy": strategy,
            "candidates": [],
            "top_k": k,
            "status": "error",
        }
