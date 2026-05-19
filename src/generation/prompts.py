from langchain_core.prompts import ChatPromptTemplate

system_prompt_template = ChatPromptTemplate.from_template("""
You are an expert assistant. Answer the following question based only on the provided context.

The context is a JSON array of retrieved documents with this format:

[
  {
    "id": str,
    "score": float,
    "content": str,
    "chunk_number": int,
    "source": str
  }
]

<context>
{context}
</context>

Question:
{question}
""")
