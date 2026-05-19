from pydantic import BaseModel


class RetrieveInfoRequest(BaseModel):
    query: str
