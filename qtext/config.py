from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal, Type

import msgspec

from qtext.ranker import CrossEncoderClient, Ranker
from qtext.schema import DefaultTable

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "qtext" / "config.json"


class ServerConfig(msgspec.Struct, kw_only=True, frozen=True):
    host: str = "0.0.0.0"
    port: Annotated[int, msgspec.Meta(ge=1, le=65535)] = 8000
    log_level: int = logging.DEBUG


class VectorStoreConfig(msgspec.Struct, kw_only=True, frozen=True):
    url: str = "postgresql://postgres:password@127.0.0.1:5432/"
    schema: Type[DefaultTable] = DefaultTable


class EmbeddingConfig(msgspec.Struct, kw_only=True, frozen=True):
    client: Literal["openai", "cohere"] = "openai"
    model_name: str = "thenlper/gte-base"
    dim: Annotated[int, msgspec.Meta(ge=1, le=65535)] = 768
    api_key: str = "fake"
    api_endpoint: str = "http://127.0.0.1:8080"
    timeout: int = 300


class SparseEmbeddingConfig(msgspec.Struct, kw_only=True, frozen=True):
    addr: str = "http://127.0.0.1:8083"
    timeout: int = 10
    dim: int = 30522


class RankConfig(msgspec.Struct, kw_only=True, frozen=True):
    ranker: Type[Ranker] = CrossEncoderClient
    params: dict[str, str] = msgspec.field(
        default_factory=lambda: {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "addr": "http://127.0.0.1:8082",
        }
    )


class HighlightConfig(msgspec.Struct, kw_only=True, frozen=True):
    addr: str = "http://127.0.0.1:8081"


class Config(msgspec.Struct, kw_only=True, frozen=True):
    server: ServerConfig = ServerConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    sparse: SparseEmbeddingConfig = SparseEmbeddingConfig()
    ranker: RankConfig = RankConfig()
    highlight: HighlightConfig = HighlightConfig()

    @classmethod
    def with_config_file(cls) -> Config:
        if not DEFAULT_CONFIG_PATH.is_file():
            return cls()
        return msgspec.json.decode(DEFAULT_CONFIG_PATH.read_bytes(), type=cls)
