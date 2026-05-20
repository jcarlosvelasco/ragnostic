import json
import logging
import os

import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.generation.prompts import system_prompt_template
from src.shared.model.RetrievedDocument import RetrievedDocument

logger = logging.getLogger(__name__)

CHAT_MODEL = "gemma4:e2b"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

llm = ChatOllama(model=CHAT_MODEL, temperature=0.2, base_url=OLLAMA_BASE_URL)
chain = system_prompt_template | llm | StrOutputParser()


async def ensure_generation_model():
    url = f"{OLLAMA_BASE_URL}/api/pull"
    data = {"model": CHAT_MODEL}
    logger.info("Pulling model %s...", CHAT_MODEL)

    async with httpx.AsyncClient() as client:
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
    retrieved_documents = [doc.model_dump() for doc in documents]

    result = await chain.ainvoke(
        {"context": json.dumps(retrieved_documents), "question": query}
    )

    return result
