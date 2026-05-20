import json
import logging
import os

import httpx
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


async def ensure_embedding_model():
    url = f"{OLLAMA_BASE_URL}/api/pull"
    data = {"model": EMBEDDING_MODEL}
    logger.info("Pulling model %s...", EMBEDDING_MODEL)

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=data) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line:
                    status = json.loads(line)
                    status_msg = status.get("status", "")

                    if "completed" in status_msg:
                        logger.info("  %s", status_msg)

    logger.info("Model %s ready", EMBEDDING_MODEL)


async def convert_to_embedding(text: str) -> list[float]:
    return await embeddings.aembed_query(text)
