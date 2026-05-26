import hashlib
import os
import random

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ingestion.model.payload import Payload
from src.shared.model.RetrievedDocument import RetrievedDocument

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
_client = AsyncQdrantClient(url=QDRANT_URL)


def get_docs_collection_name() -> str:
    return "docs_store"


async def is_store_empty(collection_name: str) -> bool:
    exists = await _client.collection_exists(collection_name)
    if not exists:
        return True

    collection = await _client.get_collection(collection_name=collection_name)
    return collection.points_count == 0


async def create_vector_store(
    collection_name: str,
    size: int,
    distance: Distance = Distance.DOT,
):
    exists = await _client.collection_exists(collection_name)
    if exists:
        return

    await _client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=size, distance=distance),
    )


def content_to_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


async def ensure_collection_exists(
    collection_name: str,
    vector_size: int,
):
    exists = await _client.collection_exists(collection_name)
    if not exists:
        await create_vector_store(collection_name, vector_size)


async def append_vectors_batch(
    collection_name: str,
    vectors: list[list[float]],
    payloads: list[Payload],
):
    points = [
        PointStruct(
            id=content_to_id(payload.content),
            vector=vector,
            payload=payload.to_dict(),
        )
        for vector, payload in zip(vectors, payloads)
    ]
    await _client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )


async def retrieve_info(
    collection_name: str,
    vector: list[float],
    n_items: int = 3,
) -> list[RetrievedDocument]:
    exists = await _client.collection_exists(collection_name)
    if not exists:
        return []

    search_result = await _client.query_points(
        collection_name=collection_name,
        query=vector,
        with_payload=True,
        limit=n_items,
    )

    return [
        RetrievedDocument(
            id=str(p.id),
            score=p.score,
            content=p.payload["content"],
            chunk_number=p.payload["chunk_number"],
            source=p.payload["source"],
        )
        for p in search_result.points
        if p.payload is not None
    ]


async def clean_vector_store(collection_name: str):
    await _client.delete_collection(collection_name=collection_name)


async def fetch_random_chunks(collection_name: str, n: int = 25) -> list[dict]:
    count_response = await _client.count(collection_name=collection_name)
    count = count_response.count

    all_points, _ = await _client.scroll(
        collection_name=collection_name,
        limit=count,
        with_payload=True,
        with_vectors=False,
    )

    sample = random.sample(all_points, min(n, len(all_points)))
    return [
        {
            "text": p.payload["content"],
            "source": p.payload["source"],
        }
        for p in sample
        if p.payload is not None
    ]
