import os
from openai import OpenAI
from pinecone import Pinecone

from .types import RetrievalResult, RetrievalRoute
from .router import route_query
from .vector_retriever import vector_search
from .bm25_retriever import bm25_search
from .reranker import rerank
from src.utils.query_logger import log_retrieval

# Candidates fetched from each retriever before re-ranking
_CANDIDATE_K = 8
# Final results returned after re-ranking
_TOP_N = 3
# Skip reranking when top vector result cosine similarity >= this threshold.
# Calibrated for NV-Embed-V1 which scores in the 0.3-0.7 range (vs ~0.7-0.95 for OpenAI embeddings).
_RERANK_CONFIDENCE_THRESHOLD = 0.65


def retrieve(
    query: str,
    top_n: int = _TOP_N,
    metadata_filter: dict | None = None,
    nvidia_client: OpenAI | None = None,
    pinecone_client: Pinecone | None = None,
) -> tuple[list[RetrievalResult], RetrievalRoute, bool]:
    """
    Main retrieval entry point.

    Returns (results, route, reranked) where:
      - route indicates BM25 or vector path
      - reranked is True if Pinecone rerank was called, False if skipped
    """
    if nvidia_client is None:
        nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
    if pinecone_client is None:
        pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    pinecone_index = pinecone_client.Index(
        os.environ.get("PINECONE_INDEX_NAME", "meridian-airlines")
    )

    route = route_query(query)

    if route == RetrievalRoute.BM25:
        candidates = bm25_search(query, top_k=_CANDIDATE_K)
        results = rerank(query, candidates, top_n=top_n, pinecone_client=pinecone_client)
        log_retrieval(route=route.value, top_score=0.0, reranked=True, result_count=len(results))
        return results, route, True

    # Vector path
    candidates = vector_search(
        query,
        top_k=_CANDIDATE_K,
        metadata_filter=metadata_filter,
        nvidia_client=nvidia_client,
        pinecone_index=pinecone_index,
    )

    top_score = candidates[0].score if candidates else 0.0

    if top_score >= _RERANK_CONFIDENCE_THRESHOLD:
        results = candidates[:top_n]
        log_retrieval(route=route.value, top_score=top_score, reranked=False, result_count=len(results))
        return results, route, False

    results = rerank(query, candidates, top_n=top_n, pinecone_client=pinecone_client)
    log_retrieval(route=route.value, top_score=top_score, reranked=True, result_count=len(results))
    return results, route, True
