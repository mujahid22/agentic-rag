import json
from pathlib import Path

import bm25s

from .types import RetrievalResult

_BM25_DIR = Path(__file__).parent.parent.parent / "data" / "bm25_index"

_retriever: bm25s.BM25 | None = None
_metadata: list[dict] | None = None


def _load() -> None:
    global _retriever, _metadata
    if _retriever is not None:
        return
    _retriever = bm25s.BM25.load(str(_BM25_DIR / "index"), load_corpus=True)
    with open(_BM25_DIR / "corpus_metadata.json", encoding="utf-8") as f:
        _metadata = json.load(f)


def bm25_search(
    query: str,
    top_k: int = 10,
) -> list[RetrievalResult]:
    _load()

    tokenized = bm25s.tokenize([query], stopwords="en")
    # Retrieve without corpus to get integer indices into _metadata
    raw_results, scores = _retriever.retrieve(tokenized, k=min(top_k, len(_metadata)))

    results = []
    for item, score in zip(raw_results[0], scores[0]):
        try:
            # bm25s 0.3.x returns dicts: {'id': <int_index>, 'text': <str>}
            idx = item["id"] if isinstance(item, dict) else int(item)
            meta = _metadata[int(idx)]
        except (IndexError, ValueError, KeyError, TypeError):
            continue
        chunk_id = meta["chunk_id"]

        results.append(RetrievalResult(
            chunk_id=chunk_id,
            text=meta.get("text", ""),
            score=float(score),
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
            retrieval_method="bm25",
        ))
    return results
