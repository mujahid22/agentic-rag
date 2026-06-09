import re
from .types import RetrievalRoute

# Patterns that signal exact-match / ID-based queries → BM25
_ID_PATTERNS = [
    re.compile(r'\bMA-[A-Z0-9]{6}\b'),       # PNR:       MA-XKPL92
    re.compile(r'\bMA-\d{3,4}\b'),            # Flight:    MA-204
    re.compile(r'\bMA-\d{13}\b'),             # e-Ticket:  MA-1762345678901
    re.compile(r'\bMA-\d{8}\b'),              # FFP acct:  MA-88234410
    re.compile(r'\bMA[A-Z]{3}\d{5}\b'),       # Bag tag:   MADXB99999
    re.compile(r'\b[A-Z]{2}\d{3,4}\b'),       # Generic flight number (e.g. LH404)
]


def route_query(query: str) -> RetrievalRoute:
    if any(p.search(query) for p in _ID_PATTERNS):
        return RetrievalRoute.BM25
    return RetrievalRoute.VECTOR
