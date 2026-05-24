from pydantic import BaseModel


class EvaluationRow(BaseModel):
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str
