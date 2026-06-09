"""
Meridian Airlines — Agentic RAG  (CLI test mode)

Usage:
    uv run python -m src.agent.run_agent
    uv run python -m src.agent.run_agent "Can I cancel my Economy Saver ticket?"
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.agent.tool_selector import select_tools
from src.agent.graph import run_query

_SAMPLE_QUERIES = [
    "What is the cancellation fee for an Economy Saver ticket?",
    "How many kilos of checked baggage can I take in Business class?",
    "My flight MA-204 was delayed by 4 hours. Am I entitled to compensation?",
    "How do I earn Meridian Miles on partner flights?",
    "Can I travel with my infant without buying a separate seat?",
]


def _interactive() -> None:
    print("=" * 60)
    print("  Meridian Airlines — AI Assistant (type 'quit' to exit)")
    print("=" * 60)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        tools_selected = select_tools(query)
        print(f"[Tools selected: {', '.join(tools_selected)}]")
        print("\nAssistant:", flush=True)
        answer = run_query(query)
        print(answer)


def _single(query: str) -> None:
    tools_selected = select_tools(query)
    print(f"Query    : {query}")
    print(f"Tools    : {', '.join(tools_selected)}")
    print("-" * 60)
    print(run_query(query))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _single(" ".join(sys.argv[1:]))
    else:
        _interactive()
