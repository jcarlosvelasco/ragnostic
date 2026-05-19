import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.schema.RetrieveInfoRequest import RetrieveInfoRequest
from src.ingestion.repository import load_docs
from src.retrieval.repository import retrive_from_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_docs()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/retrieve")
def retrieve(query: RetrieveInfoRequest):
    retrive_from_query(query.query)
    return {"info": ""}
