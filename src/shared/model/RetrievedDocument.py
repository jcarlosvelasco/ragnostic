from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    id: str
    score: float
    content: str
    chunk_number: int
    source: str
