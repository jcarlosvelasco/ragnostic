import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def getClient() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333")
    return client


client = getClient()


def create_vector_store(client: QdrantClient, collection_name: str, size: int):
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=size, distance=Distance.DOT),
    )


def content_to_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# Falta definir payload, puede tener campos text: str, source: str (nombre del documento), chunk: int (numero de chunk)


async def append_vector(
    client: QdrantClient, collection_name: str, vector: list[float], payload: dict
):
    point_id = content_to_id(payload["text"])

    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )
