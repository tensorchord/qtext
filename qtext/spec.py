from __future__ import annotations

import msgspec
import numpy as np
from reranker import Record


class SparseEmbedding(msgspec.Struct, kw_only=True, frozen=True):
    dim: int
    indices: list[int]
    values: list[float]

    def to_str(self) -> str:
        dense = np.zeros(self.dim)
        dense[self.indices] = self.values
        return f"[{','.join(map(str, dense))}]"


class QueryDocRequest(msgspec.Struct, kw_only=True):
    namespace: str
    query: str
    limit: int = 10
    vector: list[float] | None = None
    sparse_vector: SparseEmbedding | None = None
    metadata: dict | None = None

    def to_record(self) -> Record:
        return Record(
            text=self.query,
            vector=self.vector,
        )


class AddNamespaceRequest(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    vector_dim: int = 0
    sparse_vector_dim: int = 0


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
