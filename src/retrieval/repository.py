from settings import settings
from src.retrieval.reranker import rerank
from src.shared.embedder import convert_to_embedding
from src.shared.model.RetrievedDocument import RetrievedDocument
from src.shared.vector_store import get_docs_collection_name, retrieve_info


async def retrieve_from_query(query: str) -> list[RetrievedDocument]:
    vector_query = await convert_to_embedding(query)

    collection_name = get_docs_collection_name()

    response = await retrieve_info(
        collection_name, vector_query, n_items=settings.retriever_k
    )
    reranked_response = rerank(query, response, k_final=settings.reranker_k)
    return reranked_response
