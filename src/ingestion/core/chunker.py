import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.model.payload import Payload

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

logger = logging.getLogger(__name__)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_document_content(content: str, doc_file_path: str) -> list[Payload]:
    logger.info("Chunking document %s", doc_file_path)
    texts = splitter.split_text(content)
    chunks = [
        Payload(
            content=chunk,
            chunk_number=i,
            source=doc_file_path,
        )
        for i, chunk in enumerate(texts)
    ]
    logger.info("Chunking completed for %s: %d chunks", doc_file_path, len(chunks))
    return chunks
