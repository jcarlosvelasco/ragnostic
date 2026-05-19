from src.shared.embedder import convert_to_embedding, ensure_model
from src.shared.vector_store import get_client, get_docs_collection_name, retrieve_info


def retrive_from_query(query: str):
    ensure_model()

    vector_query = convert_to_embedding(query)

    collection_name = get_docs_collection_name()
    qdrant_client = get_client()

    response = retrieve_info(qdrant_client, collection_name, vector_query)
