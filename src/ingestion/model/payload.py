from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Payload:
    content: str
    chunk_number: int
    source: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)
