from __future__ import annotations

from datetime import datetime
from typing import Literal

import msgspec

DISTANCE = Literal["cosine", "inner_product", "euclidean"]

DISTANCE_TO_METHOD = {
    "euclidean": "vector_l2_ops",
    "cosine": "vector_cos_ops",
    "dot_product": "vector_dot_ops",
}

DISTANCE_TO_OP = {
    "euclidean": "<->",
    "cosine": "<=>",
    "dot_product": "<#>",
}


class AddDocRequest(msgspec.Struct, frozen=True, kw_only=True):
    namespace: str
    text: str
    doc_id: str | None = None
    vector: list[float] = msgspec.field(default_factory=list)
    title: str | None = None
    summary: str | None = None
    author: str | None = None
    updated_at: datetime | None = None
    tags: list[str] | None = None
    score: float = 1.0
    boost: float = 1.0


class DocResponse(msgspec.Struct, frozen=True, kw_only=True):
    id: int
    text: str
    vector: list[float]
    similarity: float
    score: float
    boost: float
    title: str | None
    summary: str | None
    author: str | None
    updated_at: datetime | None
    tags: list[str] | None

    @classmethod
    def from_query_result(cls, result):
        return msgspec.convert(
            {
                "id": result[0],
                "text": result[1],
                "vector": result[2].tolist(),
                "score": result[3],
                "boost": result[4],
                "title": result[5],
                "summary": result[6],
                "author": result[7],
                "updated_at": result[8],
                "tags": result[9],
                "similarity": result[10],
            },
            type=cls,
        )


class QueryDocRequest(msgspec.Struct, kw_only=True):
    namespace: str
    query: str
    limit: int = 10
    vector: list[float] | None = None
    metadata: dict | None = None
    distance: DISTANCE = "cosine"


class AddNamespaceRequest(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    vector_dim: int
    distance: DISTANCE = "cosine"
