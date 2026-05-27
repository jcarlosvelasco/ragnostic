import asyncio

from src.ingestion.repository import load_docs


async def ingest():
    await load_docs()


if __name__ == "__main__":
    asyncio.run(ingest())
