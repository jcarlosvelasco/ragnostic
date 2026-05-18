import logging
import os
from pathlib import Path

from src.docs.core.chunker import chunk_document_content
from src.docs.core.embedder import convert_to_embedding
from src.docs.core.scraper import store_docs_in_files
from src.docs.core.vector_store import append_vector, get_client, is_store_empty

logger = logging.getLogger(__name__)


def get_docs_folder_path() -> str:
    current_dir = Path(__file__).parent.parent
    docs_path = str(current_dir / "data" / "docs")
    return docs_path


async def load_docs():
    collection_name = "docs_store"
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

                logger.info("Loading document %s into store (%d chunks)", file_path, len(chunks))
                for i, chunk in enumerate(chunks):
                    embedding = convert_to_embedding(chunk.content)
                    await append_vector(
                        qdrant_client, collection_name, embedding, chunk
                    )
                    if (i + 1) % 10 == 0:
                        logger.info("  Progress: %d/%d chunks inserted", i + 1, len(chunks))

    logger.info("Ingestion complete")
