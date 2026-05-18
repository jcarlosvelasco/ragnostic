import json

import requests


def convert_to_embedding(text: str) -> list[float]:
    url = "http://ollama:11434/api/embeddings"
    data = {"model": "nomic-embed-text", "prompt": text}
    response = requests.post(url, json=data)
    return json.loads(response.text)["embedding"]
