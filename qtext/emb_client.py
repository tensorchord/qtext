from __future__ import annotations

import httpx
import msgspec
import openai

from qtext.log import logger
from qtext.metrics import embedding_histogram, sparse_histogram
from qtext.spec import SparseEmbedding
from qtext.utils import time_it


class EmbeddingClient:
    def __init__(self, model_name: str, api_key: str, endpoint: str, timeout: int):
        self.model_name = model_name
        self.client = openai.Client(
            api_key=api_key,
            base_url=endpoint or None,
            timeout=timeout,
        )

    @time_it
    @embedding_histogram.time()
    def embedding(self, text: str | list[str]) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        if len(response.data) > 1:
            return [data.embedding for data in response.data]
        return response.data[0].embedding


class SparseEmbeddingClient:
    def __init__(self, endpoint: str, dim: int, timeout: int) -> None:
        self.dim = dim
        self.client = httpx.Client(base_url=endpoint, timeout=timeout)
        self.decoder = msgspec.json.Decoder(type=list[SparseEmbedding])

    @time_it
    @sparse_histogram.time()
    def sparse_embedding(
        self, text: str | list[str]
    ) -> list[SparseEmbedding] | SparseEmbedding:
        resp = self.client.post("/inference", json=text)
        if resp.is_error:
            logger.info(
                "failed to call sparse embedding [%d]: %s",
                resp.status_code,
                resp.content,
            )
            resp.raise_for_status()
        sparse = self.decoder.decode(resp.content)

        if len(sparse) == 1:
            return sparse[0]
        return sparse
