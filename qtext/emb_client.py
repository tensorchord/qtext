from __future__ import annotations

import openai

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
    def embedding(self, text: str | list[str]) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        if len(response.data) > 1:
            return [data.embedding for data in response.data]
        return response.data[0].embedding
