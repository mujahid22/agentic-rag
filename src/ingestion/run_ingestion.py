"""
Meridian Airlines — Agentic RAG Ingestion Pipeline

Usage:
    uv run python -m src.ingestion.run_ingestion
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from .chunker import load_and_chunk_docs
from .embedder import embed_chunks
from .pinecone_uploader import upsert_chunks
from .bm25_indexer import build_and_save


def main() -> None:
    missing = [k for k in ("NVIDIA_API_KEY", "PINECONE_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"ERROR: missing environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your API keys.")
        sys.exit(1)

    print("=" * 60)
    print("  Meridian Airlines — Ingestion Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading and chunking documents...")
    chunks = load_and_chunk_docs()
    doc_count = len({c.source_doc for c in chunks})
    print(f"  {len(chunks)} chunks from {doc_count} documents")

    print("\n[2/4] Building BM25 index...")
    build_and_save(chunks)

    print("\n[3/4] Embedding chunks via NVIDIA NIM (NV-Embed-V2)...")
    embedded = embed_chunks(chunks)
    print(f"  Embedded {len(embedded)} chunks")

    print("\n[4/4] Uploading vectors to Pinecone...")
    n = upsert_chunks(embedded)
    print(f"  Upserted {n} vectors")

    print("\n" + "=" * 60)
    print("  Ingestion complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
