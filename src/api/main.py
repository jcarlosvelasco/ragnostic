from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.docs.scraper import load_docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_docs()
    yield
    # Clean up the ML models and release the resources


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
