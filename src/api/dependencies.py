import os


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")

    if provider == "mock":
        from langchain_core.language_models.fake import FakeListLLM

        return FakeListLLM(responses=["This is a mock response for CI testing."])

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
