"""
Ingest a single new document into Pinecone + rebuild BM25 index.

Usage:
    uv run python -m src.ingestion.ingest_single docs/guide_route_network.md
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from .chunker import load_and_chunk_docs, DOCS_DIR
from .embedder import embed_chunks
from .pinecone_uploader import upsert_chunks
from .bm25_indexer import build_and_save


def main(target_filename: str) -> None:
    target = Path(target_filename).name

    print(f"[1/4] Chunking new document: {target}")
    all_chunks = load_and_chunk_docs()
    new_chunks = [c for c in all_chunks if c.source_doc == target]
    if not new_chunks:
        print(f"ERROR: {target} not found in {DOCS_DIR}")
        sys.exit(1)
    print(f"  {len(new_chunks)} chunks")

    print("[2/4] Rebuilding BM25 index over full corpus...")
    build_and_save(all_chunks)
    print(f"  BM25 rebuilt over {len(all_chunks)} total chunks")

    print("[3/4] Embedding new chunks via NVIDIA NIM...")
    embedded = embed_chunks(new_chunks)
    print(f"  Embedded {len(embedded)} chunks")

    print("[4/4] Upserting new vectors to Pinecone...")
    n = upsert_chunks(embedded)
    print(f"  Upserted {n} vectors")

    print("\nDone. New document is live in the knowledge base.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m src.ingestion.ingest_single <doc_filename>")
        sys.exit(1)
    main(sys.argv[1])
