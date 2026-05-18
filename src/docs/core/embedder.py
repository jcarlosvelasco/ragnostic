import json
import logging

import requests

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"


def ensure_model():
    url = "http://ollama:11434/api/pull"
    data = {"model": EMBEDDING_MODEL}
    logger.info("Pulling model %s...", EMBEDDING_MODEL)
    response = requests.post(url, json=data, stream=True, timeout=300)
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if line:
            status = json.loads(line)
            status_msg = status.get("status", "")
            if "completed" in status_msg:
                logger.info("  %s", status_msg)
    logger.info("Model %s ready", EMBEDDING_MODEL)


def convert_to_embedding(text: str) -> list[float]:
    url = "http://ollama:11434/api/embeddings"
    data = {"model": EMBEDDING_MODEL, "prompt": text}
    response = requests.post(url, json=data)
    body = json.loads(response.text)
    if "error" in body:
        raise RuntimeError(f"Ollama error: {body['error']}")
    return body["embedding"]
