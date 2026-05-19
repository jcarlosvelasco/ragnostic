import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ingestion.model.payload import Payload
from src.retrieval.model.RetrievedDocument import RetrievedDocument

_client = QdrantClient(url="http://qdrant:6333")


def get_client() -> QdrantClient:
    return _client


def get_docs_collection_name() -> str:
    return "docs_store"


def is_store_empty(client: QdrantClient, collection_name: str) -> bool:
    if not client.collection_exists(collection_name):
        return True

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


def retrieve_info(
    client: QdrantClient,
    collection_name: str,
    vector: list[float],
    n_items: int = 3,
) -> list[RetrievedDocument]:
    if not client.collection_exists(collection_name):
        return []

    search_result = client.query_points(
        collection_name=collection_name, query=vector, with_payload=True, limit=n_items
    ).points

    return [
        RetrievedDocument(
            id=str(p.id),
            score=p.score,
            content=p.payload["content"],
            chunk_number=p.payload["chunk_number"],
            source=p.payload["source"],
        )
        for p in search_result
        if p.payload is not None
    ]
