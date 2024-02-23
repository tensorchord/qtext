from __future__ import annotations

from datetime import datetime
from typing import Literal

import msgspec


class AddDocRequest(msgspec.Struct, frozen=True, kw_only=True):
    namespace: str
    text: str
    doc_id: str | None = None
    vector: list[float]
    title: str | None = None
    summary: str | None = None
    author: str | None = None
    updated_at: datetime | None = None
    tags: list[str] | None = None
    score: float = 1.0
    boost: float = 1.0


class QueryDocRequest(msgspec.Struct, kw_only=True):
    namespace: str
    query: str
    limit: int = 10
    vector: list[float] | None = None
    metadata: dict | None = None


class AddNamespaceRequest(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    vector_dim: int
    distance: Literal["cosine", "inner_product", "euclidean"] = "cosine"
