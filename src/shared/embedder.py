import json
import logging

import httpx

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"


async def ensure_embedding_model():
    url = "http://ollama:11434/api/pull"
    data = {"model": EMBEDDING_MODEL}
    logger.info("Pulling model %s...", EMBEDDING_MODEL)

    async with httpx.AsyncClient(timeout=300) as client:
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
    url = "http://ollama:11434/api/embeddings"
    data = {"model": EMBEDDING_MODEL, "prompt": text}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        body = json.loads(response.text)

        if "error" in body:
            raise RuntimeError(f"Ollama error: {body['error']}")

    return body["embedding"]
