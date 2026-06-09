import os
import time
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from .chunker import Chunk
from .embedder import EMBED_DIM

UPSERT_BATCH = 100


def _get_or_create_index(pc: Pinecone, index_name: str):
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        print(f"  Creating Pinecone index '{index_name}' (dim={EMBED_DIM}, metric=cosine)...")
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
    return pc.Index(index_name)


def upsert_chunks(
    embedded: list[tuple[Chunk, list[float]]],
    index_name: str | None = None,
) -> int:
    index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "meridian-airlines")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = _get_or_create_index(pc, index_name)

    total_upserted = 0
    for i in tqdm(range(0, len(embedded), UPSERT_BATCH), desc="Upserting to Pinecone"):
        batch = embedded[i : i + UPSERT_BATCH]
        vectors = [
            {
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": chunk.to_pinecone_metadata(),
            }
            for chunk, embedding in batch
        ]
        index.upsert(vectors=vectors)
        total_upserted += len(vectors)

    return total_upserted
