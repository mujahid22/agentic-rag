import re
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict

from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


@dataclass
class Chunk:
    chunk_id: str
    source_doc: str
    text: str
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

    def to_pinecone_metadata(self) -> dict:
        d = asdict(self)
        d.pop("chunk_id")
        return d


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}, content
    meta = yaml.safe_load(m.group(1)) or {}
    return meta, content[m.end():]


def load_and_chunk_docs(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks: list[Chunk] = []

    for md_file in sorted(docs_dir.glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        texts = splitter.split_text(body)
        total = len(texts)

        for i, text in enumerate(texts):
            chunks.append(Chunk(
                chunk_id=f"{md_file.stem}_chunk_{i:04d}",
                source_doc=md_file.name,
                text=text.strip(),
                doc_type=str(meta.get("doc_type", "")),
                topic=str(meta.get("topic", "")),
                subtopic=str(meta.get("subtopic", "")),
                section=str(meta.get("section", "")),
                fare_class=str(meta.get("fare_class", "")),
                route_type=str(meta.get("route_type", "")),
                passenger_type=str(meta.get("passenger_type", "")),
                applies_to=str(meta.get("applies_to", "")),
                effective_date=str(meta.get("effective_date", "")),
                tier_level=str(meta.get("tier_level", "")),
                chunk_index=i,
                total_chunks=total,
            ))

    return chunks
