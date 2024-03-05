from __future__ import annotations

from pathlib import Path
from typing import Annotated, Type

import msgspec
from reranker import HybridRanker, Ranker

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "qtext" / "config.json"


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


class RankConfig(msgspec.Struct, kw_only=True, frozen=True):
    ranker: Type[Ranker] = HybridRanker
    params: dict[str, str] = msgspec.field(default_factory=dict)


class HighlightConfig(msgspec.Struct, kw_only=True, frozen=True):
    addr: str = "http://127.0.0.1:8081"


class Config(msgspec.Struct, kw_only=True, frozen=True):
    server: ServerConfig = ServerConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    ranker: RankConfig = RankConfig()
    highlight: HighlightConfig = HighlightConfig()

    @classmethod
    def with_config_file(cls) -> Config:
        if not DEFAULT_CONFIG_PATH.is_file():
            return cls()
        return msgspec.json.decode(DEFAULT_CONFIG_PATH.read_bytes(), type=cls)
