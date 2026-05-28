import json
import logging
import os

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from settings import settings
from src.generation.prompts import system_prompt_template
from src.shared.langfuse import langfuse_handler
from src.shared.model.RetrievedDocument import RetrievedDocument

logger = logging.getLogger(__name__)

CHAT_MODEL = settings.generator_model

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

provider = os.getenv("LLM_PROVIDER", "ollama")


def get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "mock":
        return GenericFakeChatModel(
            messages=iter(
                [AIMessage(content="This is a mock response for CI testing.")]
            )
        )
    return ChatOllama(model=CHAT_MODEL, temperature=0.2, base_url=OLLAMA_BASE_URL)


llm = get_llm()

chain = (system_prompt_template | llm | StrOutputParser()).with_config(
    run_name="Generate response with provided context"
)


async def ensure_generation_model():
    if provider == "mock":
        return
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


class GenerateResponse(BaseModel):
    response: str


async def generate_response(
    query: str, documents: list[RetrievedDocument]
) -> GenerateResponse:
    retrieved_documents = [doc.model_dump() for doc in documents]

    result = await chain.ainvoke(
        {"context": json.dumps(retrieved_documents), "question": query},
        config={
            "callbacks": [langfuse_handler],
            "metadata": {
                "query": query,
                "num_documents": len(documents),
            },
        },
    )

    return GenerateResponse(response=result)
