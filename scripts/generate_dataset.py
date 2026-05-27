import asyncio
import json
import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.shared.vector_store import (
    fetch_random_chunks,
    get_docs_collection_name,
)

CHAT_MODEL = "gemma4:e2b"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model=CHAT_MODEL, temperature=0.2, base_url=OLLAMA_BASE_URL)

PROMPT = """Given the following LangChain documentation excerpt, generate 2 questions
whose answers are contained ONLY within the text. The questions should be
specific and technical, like the kind a developer would ask.

Respond ONLY with valid JSON, without markdown or explanations:
[
  {{"question": "...", "ground_truth": "..."}},
  {{"question": "...", "ground_truth": "..."}}
]

Excerpt:
{chunk}
"""


def is_valid_chunk(text: str) -> bool:
    if len(text.strip()) < 150:
        return False

    if re.search(r"\d+\.\d+,\d+\.\d+", text):
        return False

    tag_count = len(re.findall(r"<[A-Za-z]", text))
    if tag_count > 4:
        return False

    yaml_lines = len(re.findall(r"^\s{2,}\w+:", text, re.MULTILINE))
    if yaml_lines > 5:
        return False

    code_lines = len(re.findall(r"^(import |from |pip install)", text, re.MULTILINE))
    words = len(text.split())
    if code_lines > 2 and words < 40:
        return False

    return True


async def generate_dataset(n_chunks: int = 25):
    collection_name = get_docs_collection_name()
    chunks = await fetch_random_chunks(collection_name, n_chunks)
    dataset = []

    for i, chunk in enumerate(chunks):
        if not is_valid_chunk(chunk["text"]):
            print(f"  ⏭️  Chunk {i + 1} filtered (not useful)")
            continue

        print(f"Generating questions for chunk {i + 1}/{len(chunks)}...")
        try:
            chain = llm | StrOutputParser()

            response = await chain.ainvoke(PROMPT.format(chunk=chunk["text"]))

            print(f"Response: {response}")

            content = response.replace("```json", "").replace("```", "").strip()
            pairs = json.loads(content)
            for pair in pairs:
                pair["source"] = chunk["source"]
                pair["chunk"] = chunk["text"]
            dataset.extend(pairs)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error in chunk {i + 1}: {e} — skipping")
            continue

    output_path = "evals/golden_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated dataset: {len(dataset)} pairs in {output_path}")
    return dataset


if __name__ == "__main__":
    asyncio.run(generate_dataset())
