from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    generator_model: str
    chunk_size: int
    chunk_overlap: int
    chunk_min_size: int
    embedding_model: str
    eval_model: str
    eval_embedding_model: str
    retriever_k: int
    reranker_k: int


settings = Settings(
    generator_model="gemma4:e2b-mlx",
    chunk_size=700,
    chunk_overlap=100,
    chunk_min_size=100,
    embedding_model="nomic-embed-text",
    eval_model="gemma4:e2b-mlx",
    eval_embedding_model="nomic-embed-text",
    retriever_k=10,
    reranker_k=3,
)
