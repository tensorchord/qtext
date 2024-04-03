from __future__ import annotations

import httpx
import msgspec

from qtext.spec import AddNamespaceRequest, QueryDocRequest, QueryExplainResponse


class QTextClient:
    def __init__(self, addr: str = "http://127.0.0.1", port: int = 8000) -> None:
        self.client = httpx.Client(base_url=f"{addr}:{port}/api/")
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(QueryExplainResponse)

    def add_namespace(
        self, namespace: str, vector_dim: int = 0, sparse_dim: int = 0
    ) -> None:
        self.client.post(
            "/namespace",
            content=self.encoder.encode(
                AddNamespaceRequest(
                    name=namespace,
                    vector_dim=vector_dim,
                    sparse_vector_dim=sparse_dim,
                )
            ),
        )

    def query(self, namespace: str, query: str) -> dict:
        return self.client.post(
            "/query",
            content=self.encoder.encode(
                QueryDocRequest(
                    namespace=namespace,
                    query=query,
                )
            ),
        ).json()

    def query_explain(self, namespace: str, query: str) -> QueryExplainResponse:
        resp = self.client.post(
            "/query_explain",
            content=self.encoder.encode(
                QueryDocRequest(
                    namespace=namespace,
                    query=query,
                )
            ),
        )
        resp.raise_for_status()
        return self.decoder.decode(resp.content)


class QTextAsyncClient:
    def __init__(self, addr: str = "http://127.0.0.1", port: int = 8000) -> None:
        self.client = httpx.AsyncClient(base_url=f"{addr}:{port}/api/")
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(QueryExplainResponse)

    async def add_namespace(
        self, namespace: str, vector_dim: int = 0, sparse_dim: int = 0
    ) -> None:
        await self.client.post(
            "/namespace",
            content=self.encoder.encode(
                AddNamespaceRequest(
                    name=namespace,
                    vector_dim=vector_dim,
                    sparse_vector_dim=sparse_dim,
                )
            ),
        )

    async def query(self, namespace: str, query: str) -> dict:
        return await self.client.post(
            "/query",
            content=self.encoder.encode(
                QueryDocRequest(
                    namespace=namespace,
                    query=query,
                )
            ),
        ).json()

    async def query_explain(self, namespace: str, query: str) -> QueryExplainResponse:
        resp = await self.client.post(
            "/query_explain",
            content=self.encoder.encode(
                QueryDocRequest(
                    namespace=namespace,
                    query=query,
                )
            ),
        )
        resp.raise_for_status()
        return self.decoder.decode(resp.content)
