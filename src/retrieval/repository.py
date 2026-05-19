from src.shared.embedder import convert_to_embedding
from src.shared.model.RetrievedDocument import RetrievedDocument
from src.shared.vector_store import get_client, get_docs_collection_name, retrieve_info


async def retrieve_from_query(query: str) -> list[RetrievedDocument]:
    vector_query = await convert_to_embedding(query)

    collection_name = get_docs_collection_name()
    qdrant_client = get_client()

    response = await retrieve_info(qdrant_client, collection_name, vector_query)
    return response
