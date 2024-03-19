from __future__ import annotations

import msgspec
from reranker import Record


class QueryDocRequest(msgspec.Struct, kw_only=True):
    namespace: str
    query: str
    limit: int = 10
    vector: list[float] | None = None
    metadata: dict | None = None

    def to_record(self) -> Record:
        return Record(
            text=self.query,
            vector=self.vector,
        )


class AddNamespaceRequest(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    vector_dim: int


class HighlightRequest(msgspec.Struct, kw_only=True, frozen=True):
    query: str
    docs: list[str]
    threshold: float = 0.8
    ignore_stopwords: bool = True
    template: str = "<mark>{}</mark>"


class HighlightResponse(msgspec.Struct, kw_only=True):
    highlighted: list[str]


class HighlightScore(msgspec.Struct, kw_only=True, frozen=True):
    text: str
    score: float
