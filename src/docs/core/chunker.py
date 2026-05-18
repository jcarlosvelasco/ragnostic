from src.docs.model.payload import Payload

CHUNK_SIZE = 200
CHUNK_OVERLAP = 20


def chunk_document_content(content: str, doc_file_path: str) -> list[Payload]:
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
    return chunks
