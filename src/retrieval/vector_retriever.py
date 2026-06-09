import os
from openai import OpenAI
from pinecone import Pinecone

from .types import RetrievalResult

_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
_EMBED_MODEL = "nvidia/nv-embed-v1"


def _embed_query(query: str, client: OpenAI) -> list[float]:
    resp = client.embeddings.create(
        model=_EMBED_MODEL,
        input=[query],
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"},
    )
    return resp.data[0].embedding


def vector_search(
    query: str,
    top_k: int = 10,
    metadata_filter: dict | None = None,
    nvidia_client: OpenAI | None = None,
    pinecone_index=None,
) -> list[RetrievalResult]:
    if nvidia_client is None:
        nvidia_client = OpenAI(
            base_url=_NVIDIA_BASE_URL,
            api_key=os.environ["NVIDIA_API_KEY"],
        )
    if pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        pinecone_index = pc.Index(
            os.environ.get("PINECONE_INDEX_NAME", "meridian-airlines")
        )

    embedding = _embed_query(query, nvidia_client)

    response = pinecone_index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter,
    )

    results = []
    for match in response.matches:
        meta = match.metadata or {}
        results.append(RetrievalResult(
            chunk_id=match.id,
            text=meta.get("text", ""),
            score=float(match.score),
            source_doc=meta.get("source_doc", ""),
            doc_type=meta.get("doc_type", ""),
            topic=meta.get("topic", ""),
            subtopic=meta.get("subtopic", ""),
            section=meta.get("section", ""),
            fare_class=meta.get("fare_class", ""),
            route_type=meta.get("route_type", ""),
            passenger_type=meta.get("passenger_type", ""),
            applies_to=meta.get("applies_to", ""),
            effective_date=meta.get("effective_date", ""),
            tier_level=meta.get("tier_level", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            total_chunks=int(meta.get("total_chunks", 0)),
            retrieval_method="vector",
        ))
    return results
