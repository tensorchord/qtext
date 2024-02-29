from __future__ import annotations

from typing import Annotated

import msgspec


class ServerConfig(msgspec.Struct, kw_only=True, frozen=True):
    host: str = "0.0.0.0"
    port: Annotated[int, msgspec.Meta(ge=1, le=65535)] = 8000


class VectorStoreConfig(msgspec.Struct, kw_only=True, frozen=True):
    url: str = "postgresql://postgres:password@127.0.0.1:5432/"


class EmbeddingConfig(msgspec.Struct, kw_only=True, frozen=True):
    model_name: str = "thenlper/gte-base"
    dim: Annotated[int, msgspec.Meta(ge=1, le=65535)] = 768
    api_key: str = "fake"
    api_endpoint: str = "http://127.0.0.1:8080"
    timeout: int = 300


class Config(msgspec.Struct, kw_only=True, frozen=True):
    server: ServerConfig = ServerConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
