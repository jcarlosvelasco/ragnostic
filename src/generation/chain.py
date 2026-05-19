import json
import logging

import httpx

from src.generation.prompts import system_prompt_template
from src.shared.model.RetrievedDocument import RetrievedDocument

logger = logging.getLogger(__name__)

CHAT_MODEL = "gemma4:e2b"


async def ensure_generation_model():
    url = "http://ollama:11434/api/pull"
    data = {"model": CHAT_MODEL}
    logger.info("Pulling model %s...", CHAT_MODEL)

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", url, json=data) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line:
                    status = json.loads(line)
                    status_msg = status.get("status", "")

                    if "completed" in status_msg:
                        logger.info("  %s", status_msg)

    logger.info("Model %s ready", CHAT_MODEL)


async def generate_response(query: str, documents: list[RetrievedDocument]) -> str:
    url = "http://ollama:11434/api/generate"

    retrieved_documents = [doc.model_dump() for doc in documents]

    prompt_value = system_prompt_template.invoke(
        {
            "context": json.dumps(retrieved_documents),
            "question": query,
        }
    )

    data = {"model": CHAT_MODEL, "prompt": prompt_value}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        body = json.loads(response.text)

        if "error" in body:
            raise RuntimeError(f"Ollama error: {body['error']}")

    return body["response"]
