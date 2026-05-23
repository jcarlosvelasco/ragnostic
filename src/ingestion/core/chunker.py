import logging

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.ingestion.core.chunk_utils import clean_mdx
from src.ingestion.model.payload import Payload

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

logger = logging.getLogger(__name__)

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ],
    strip_headers=False,
)

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n```",
        "\n\n",
        "\n- ",
        "\n* ",
        "\n",
        ". ",
        " ",
        "",
    ],
)

MIN_CHUNK_SIZE = 100


def merge_chunks(sub_chunks: list[Document]) -> list[Document]:
    merged = []

    for doc in sub_chunks:
        if merged and len(doc.page_content) < MIN_CHUNK_SIZE:
            merged[-1].page_content += "\n\n" + doc.page_content
        else:
            merged.append(doc)

    return merged


def chunk_document_content(content: str, doc_file_path: str) -> list[Payload]:
    logger.info("Chunking document %s", doc_file_path)
    # logger.info("Document content: %s", content)

    res = clean_mdx(content)

    header_chunks = header_splitter.split_text(res)

    texts: list[Document] = []
    for doc in header_chunks:
        sub_chunks = char_splitter.split_documents([doc])

        texts.extend(sub_chunks)

    texts = merge_chunks(texts)

    chunks = [
        Payload(
            content=chunk.page_content,
            chunk_number=i,
            source=doc_file_path,
            metadata=chunk.metadata,
        )
        for i, chunk in enumerate(texts)
    ]

    # logger.info("Chunks: %s", chunks)

    logger.info("Chunking completed for %s: %d chunks", doc_file_path, len(chunks))
    return chunks
