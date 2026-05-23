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


async def _embed_batch(chunks: list, sem: asyncio.Semaphore) -> list:
    async def embed(chunk):
        async with sem:
            return await convert_to_embedding(chunk.content)

    return await asyncio.gather(*[embed(c) for c in chunks])


async def _iter_chunks(docs_folder_path: str):
    for root, _, files in os.walk(docs_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()
            chunks = chunk_document_content(content, file_path)
            logger.info("Chunked %s (%d chunks)", file_path, len(chunks))
            for chunk in chunks:
                yield chunk


async def load_docs():
    collection_name = get_docs_collection_name()
    is_empty = await is_store_empty(collection_name)
    if not is_empty:
        logger.info("Docs already loaded in store, skipping ingestion")
        return

    logger.info("Store is empty, starting ingestion...")
    await store_docs_in_files()

    docs_folder_path = get_docs_folder_path()
    sem = asyncio.Semaphore(EMBEDDING_CONCURRENCY)
    collection_initialized = False
    total_stored = 0
    batch: list = []

    async def flush_batch(batch: list):
        nonlocal collection_initialized, total_stored

        embeddings = await _embed_batch(batch, sem)

        if not collection_initialized:
            await ensure_collection_exists(collection_name, len(embeddings[0]))
            collection_initialized = True

        await append_vectors_batch(collection_name, embeddings, batch)
        total_stored += len(batch)
        logger.info("Progress: %d chunks stored so far", total_stored)

    logger.info("Starting streaming ingestion...")
    async for chunk in _iter_chunks(docs_folder_path):
        batch.append(chunk)
        if len(batch) >= BATCH_SIZE:
            await flush_batch(batch)
            batch = []

    if batch:
        await flush_batch(batch)

    if total_stored == 0:
        logger.info("No chunks found, skipping ingestion")
        return

    logger.info("Ingestion complete (%d chunks)", total_stored)


async def clean_all_data():
    data_path = get_docs_folder_path()
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
    logger.info("All data cleaned from %s", data_path)
    collection_name = get_docs_collection_name()
    await clean_vector_store(collection_name)
    logger.info("Vector store cleaned: %s", collection_name)
