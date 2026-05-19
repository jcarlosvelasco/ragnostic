import logging

from src.ingestion.model.payload import Payload

CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

logger = logging.getLogger(__name__)


def chunk_document_content(content: str, doc_file_path: str) -> list[Payload]:
    logger.info("Chunking document %s", doc_file_path)
    chunks = []
    for i in range(0, len(content), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = content[i : i + CHUNK_SIZE]
        chunks.append(
            Payload(
                content=chunk,
                chunk_number=i // (CHUNK_SIZE - CHUNK_OVERLAP),
                source=doc_file_path,
            )
        )
    logger.info("Chunking completed for %s: %d chunks", doc_file_path, len(chunks))
    return chunks
