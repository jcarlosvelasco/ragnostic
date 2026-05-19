import hashlib

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ingestion.model.payload import Payload
from src.shared.model.RetrievedDocument import RetrievedDocument

_client = AsyncQdrantClient(url="http://qdrant:6333")


def get_client() -> AsyncQdrantClient:
    return _client


def get_docs_collection_name() -> str:
    return "docs_store"


async def is_store_empty(client: AsyncQdrantClient, collection_name: str) -> bool:
    exists = await client.collection_exists(collection_name)
    if not exists:
        return True

    collection = await client.get_collection(collection_name=collection_name)
    return collection.points_count == 0


async def create_vector_store(
    client: AsyncQdrantClient,
    collection_name: str,
    size: int,
    distance: Distance = Distance.DOT,
):
    exists = await client.collection_exists(collection_name)
    if not exists:
        return

    await client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=size, distance=distance),
    )


def content_to_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


async def append_vector(
    client: AsyncQdrantClient,
    collection_name: str,
    vector: list[float],
    payload: Payload,
):
    is_empty = await is_store_empty(client, collection_name)
    if is_empty:
        await create_vector_store(client, collection_name, len(vector))

    point_id = content_to_id(payload.content)

    await client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[PointStruct(id=point_id, vector=vector, payload=payload.to_dict())],
    )


async def retrieve_info(
    client: AsyncQdrantClient,
    collection_name: str,
    vector: list[float],
    n_items: int = 3,
) -> list[RetrievedDocument]:
    exists = await client.collection_exists(collection_name)
    if not exists:
        return []

    search_result = await client.query_points(
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


async def clean_vector_store(client: AsyncQdrantClient, collection_name: str):
    await client.delete_collection(collection_name=collection_name)
