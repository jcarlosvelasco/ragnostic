import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.schema.RetrieveInfoRequest import RetrieveInfoRequest
from src.generation.chain import ensure_generation_model, generate_response
from src.retrieval.repository import retrieve_from_query
from src.shared.embedder import ensure_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_generation_model()
    await ensure_embedding_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/retrieve")
async def retrieve(query: RetrieveInfoRequest):
    response = await retrieve_from_query(query.query)
    result = await generate_response(query.query, response)
    print(result)
    return {"info": result}
