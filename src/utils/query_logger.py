import json
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from threading import Lock

_LOG_PATH = Path(__file__).parent.parent.parent / "logs" / "query_log.jsonl"
_lock = Lock()

# Shared across all modules in the same thread via contextvars
_current_query_id: ContextVar[str] = ContextVar("current_query_id", default="")
_query_start_time: ContextVar[float] = ContextVar("query_start_time", default=0.0)


def start_query(query: str) -> str:
    qid = uuid.uuid4().hex[:8]
    _current_query_id.set(qid)
    _query_start_time.set(time.time())
    _append({
        "query_id": qid,
        "event": "query_start",
        "timestamp": _now(),
        "query": query,
    })
    return qid


def log_tool_selection(tools: list[str]) -> None:
    qid = _current_query_id.get()
    if not qid:
        return
    _append({
        "query_id": qid,
        "event": "tool_selection",
        "timestamp": _now(),
        "tools": tools,
    })


def log_retrieval(
    route: str,
    top_score: float,
    reranked: bool,
    result_count: int,
    tool_name: str = "",
) -> None:
    qid = _current_query_id.get()
    if not qid:
        return
    _append({
        "query_id": qid,
        "event": "retrieval",
        "timestamp": _now(),
        "tool": tool_name,
        "route": route,
        "top_score": round(top_score, 4),
        "reranked": reranked,
        "results_returned": result_count,
    })


def get_current_query_id() -> str:
    return _current_query_id.get()


def set_current_query_id(qid: str) -> None:
    _current_query_id.set(qid)


def end_query(usage: dict | None = None) -> None:
    qid = _current_query_id.get()
    if not qid:
        return
    elapsed = round(time.time() - _query_start_time.get(), 2)
    record = {
        "query_id": qid,
        "event": "query_end",
        "timestamp": _now(),
        "response_time_s": elapsed,
    }
    if usage:
        record["input_tokens"] = usage.get("input_tokens")
        record["output_tokens"] = usage.get("output_tokens")
        record["total_tokens"] = usage.get("total_tokens")
    _append(record)


def load_queries(since: str | None = None) -> list[dict]:
    """Return one summary dict per query, newest first. Pass since= (timestamp string) to filter to current session."""
    if not _LOG_PATH.exists():
        return []
    records: list[dict] = []
    with _LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Group by query_id preserving insertion order
    queries: dict[str, dict] = {}
    for r in records:
        qid = r["query_id"]
        if qid not in queries:
            queries[qid] = {
                "query_id": qid,
                "query": "",
                "timestamp": "",
                "tools": [],
                "retrievals": [],
                "response_time_s": None,
                "total_tokens": None,
            }
        if r["event"] == "query_start":
            queries[qid]["query"] = r["query"]
            queries[qid]["timestamp"] = r["timestamp"]
        elif r["event"] == "tool_selection":
            queries[qid]["tools"] = r["tools"]
        elif r["event"] == "retrieval":
            queries[qid]["retrievals"].append(r)
        elif r["event"] == "query_end":
            queries[qid]["response_time_s"] = r["response_time_s"]
            queries[qid]["total_tokens"] = r.get("total_tokens")

    result = list(reversed(list(queries.values())))
    if since:
        result = [q for q in result if q["timestamp"] >= since]
    return result


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append(record: dict) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
