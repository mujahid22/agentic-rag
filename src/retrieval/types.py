from dataclasses import dataclass
from enum import Enum


class RetrievalRoute(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    source_doc: str
    doc_type: str = ""
    topic: str = ""
    subtopic: str = ""
    section: str = ""
    fare_class: str = ""
    route_type: str = ""
    passenger_type: str = ""
    applies_to: str = ""
    effective_date: str = ""
    tier_level: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    retrieval_method: str = ""
