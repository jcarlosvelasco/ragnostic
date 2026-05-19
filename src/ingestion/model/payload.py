from dataclasses import asdict, dataclass


@dataclass
class Payload:
    content: str
    chunk_number: int
    source: str

    def to_dict(self) -> dict:
        return asdict(self)
