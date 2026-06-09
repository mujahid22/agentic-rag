import os
from pinecone import Pinecone

from .types import RetrievalResult

_RERANK_MODEL = "pinecone-rerank-v0"


def rerank(
    query: str,
    results: list[RetrievalResult],
    top_n: int = 5,
    pinecone_client: Pinecone | None = None,
) -> list[RetrievalResult]:
    if not results:
        return []

    if pinecone_client is None:
        pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    documents = [r.text for r in results]

    reranked = pinecone_client.inference.rerank(
        model=_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False,
    )

    reranked_results = []
    for item in reranked.data:
        original = results[item.index]
        original.score = item.score
        reranked_results.append(original)

    return reranked_results
