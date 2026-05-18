from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.docs.repository import load_docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_docs()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}
