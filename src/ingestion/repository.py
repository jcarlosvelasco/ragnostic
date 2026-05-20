import asyncio
import logging
import os
from pathlib import Path

from src.ingestion.core.chunker import chunk_document_content
from src.ingestion.core.scraper import store_docs_in_files
from src.shared.embedder import convert_to_embedding
from src.shared.vector_store import (
    append_vectors_batch,
    clean_vector_store,
    ensure_collection_exists,
    get_client,
    get_docs_collection_name,
    is_store_empty,
)

logger = logging.getLogger(__name__)


def get_docs_folder_path() -> str:
    current_dir = Path(__file__).parent.parent
    docs_path = str(current_dir / "data" / "docs")
    return docs_path


EMBEDDING_CONCURRENCY = 20
BATCH_SIZE = 100


async def load_docs():
    collection_name = get_docs_collection_name()
    qdrant_client = get_client()
    is_empty = await is_store_empty(qdrant_client, collection_name)
    if not is_empty:
        logger.info("Docs already loaded in store, skipping ingestion")
        return

    logger.info("Store is empty, starting ingestion...")
    await store_docs_in_files()

    docs_folder_path = get_docs_folder_path()
    all_chunks = []
    for root, _, files in os.walk(docs_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()
            chunks = chunk_document_content(content, file_path)
            logger.info("Queued %s (%d chunks)", file_path, len(chunks))
            all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("No chunks found, skipping ingestion")
        return

    total = len(all_chunks)
    logger.info("Total chunks to ingest: %d", total)

    sem = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    async def embed(chunk):
        async with sem:
            return await convert_to_embedding(chunk.content)

    logger.info("Generating embeddings...")
    embeddings = await asyncio.gather(*[embed(c) for c in all_chunks])

    await ensure_collection_exists(qdrant_client, collection_name, len(embeddings[0]))

    logger.info("Storing vectors in Qdrant...")
    for i in range(0, total, BATCH_SIZE):
        batch_chunks = all_chunks[i : i + BATCH_SIZE]
        batch_embeddings = embeddings[i : i + BATCH_SIZE]
        await append_vectors_batch(
            qdrant_client, collection_name, batch_embeddings, batch_chunks
        )
        logger.info(
            "  Progress: %d/%d chunks stored", min(i + BATCH_SIZE, total), total
        )

    logger.info("Ingestion complete (%d chunks)", total)


async def clean_all_data():
    data_path = get_docs_folder_path()
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
    logger.info("All data cleaned from %s", data_path)

    collection_name = get_docs_collection_name()
    qdrant_client = get_client()

    await clean_vector_store(qdrant_client, collection_name)
    logger.info("Vector store cleaned: %s", collection_name)
