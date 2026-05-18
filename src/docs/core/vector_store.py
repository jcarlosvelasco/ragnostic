import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.docs.model.payload import Payload


def get_client() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333")
    return client


def is_store_empty(client: QdrantClient, collection_name: str) -> bool:
    if not client.collection_exists(collection_name):
        return False

    collection = client.get_collection(collection_name=collection_name)
    return collection.points_count == 0


def create_vector_store(
    client: QdrantClient,
    collection_name: str,
    size: int,
    distance: Distance = Distance.DOT,
):
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=size, distance=distance),
    )


def content_to_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


async def append_vector(
    client: QdrantClient, collection_name: str, vector: list[float], payload: Payload
):
    if is_store_empty(client, collection_name):
        create_vector_store(client, collection_name, len(vector))

    point_id = content_to_id(payload.content)

    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[PointStruct(id=point_id, vector=vector, payload=payload.to_dict())],
    )
