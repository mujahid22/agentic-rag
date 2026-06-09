import os
from openai import OpenAI
from tqdm import tqdm

from .chunker import Chunk

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embed-v1"
EMBED_DIM = 4096
BATCH_SIZE = 32


def get_client() -> OpenAI:
    return OpenAI(base_url=NVIDIA_NIM_BASE_URL, api_key=os.environ["NVIDIA_API_KEY"])


def embed_chunks(
    chunks: list[Chunk],
    client: OpenAI | None = None,
) -> list[tuple[Chunk, list[float]]]:
    if client is None:
        client = get_client()

    results: list[tuple[Chunk, list[float]]] = []

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch = chunks[i : i + BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=[c.text for c in batch],
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"},
        )
        for chunk, emb_obj in zip(batch, response.data):
            results.append((chunk, emb_obj.embedding))

    return results
