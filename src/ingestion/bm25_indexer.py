import json
from dataclasses import asdict
from pathlib import Path

import bm25s

from .chunker import Chunk

BM25_DIR = Path(__file__).parent.parent.parent / "data" / "bm25_index"


def build_and_save(chunks: list[Chunk], output_dir: Path = BM25_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = [c.text for c in chunks]
    tokenized = bm25s.tokenize(corpus, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(tokenized)
    retriever.save(str(output_dir / "index"), corpus=corpus)

    with open(output_dir / "corpus_metadata.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)

    print(f"  BM25 index saved -> {output_dir} ({len(chunks)} chunks)")


def load(index_dir: Path = BM25_DIR) -> tuple[bm25s.BM25, list[dict]]:
    retriever = bm25s.BM25.load(str(index_dir / "index"), load_corpus=True)
    with open(index_dir / "corpus_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return retriever, metadata
