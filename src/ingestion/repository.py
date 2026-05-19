import logging
import os
from pathlib import Path

from src.ingestion.core.chunker import chunk_document_content
from src.ingestion.core.scraper import store_docs_in_files
from src.shared.embedder import convert_to_embedding
from src.shared.vector_store import (
    append_vector,
    clean_vector_store,
    get_client,
    get_docs_collection_name,
    is_store_empty,
)

logger = logging.getLogger(__name__)


def get_docs_folder_path() -> str:
    current_dir = Path(__file__).parent.parent
    docs_path = str(current_dir / "data" / "docs")
    return docs_path


async def load_docs():
    collection_name = get_docs_collection_name()
    qdrant_client = get_client()

    if not is_store_empty(qdrant_client, collection_name):
        logger.info("Docs already loaded in store, skipping ingestion")
        return

    logger.info("Store is empty, starting ingestion...")

    await store_docs_in_files()

    docs_folder_path = get_docs_folder_path()

    for root, _, files in os.walk(docs_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()
                chunks = chunk_document_content(content, file_path)

                logger.info(
                    "Loading document %s into store (%d chunks)", file_path, len(chunks)
                )
                for i, chunk in enumerate(chunks):
                    embedding = await convert_to_embedding(chunk.content)
                    await append_vector(
                        qdrant_client, collection_name, embedding, chunk
                    )
                    if (i + 1) % 10 == 0:
                        logger.info(
                            "  Progress: %d/%d chunks inserted", i + 1, len(chunks)
                        )

    logger.info("Ingestion complete")


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
