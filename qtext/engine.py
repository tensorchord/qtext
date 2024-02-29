from __future__ import annotations

from qtext.config import Config
from qtext.emb_client import EmbeddingClient
from qtext.pg_client import PgVectorsClient
from qtext.spec import AddDocRequest, AddNamespaceRequest, DocResponse, QueryDocRequest


class RetrievalEngine:
    def __init__(self, config: Config) -> None:
        self.pg_client = PgVectorsClient(config.vector_store.url)
        self.emb_client = EmbeddingClient(
            model_name=config.embedding.model_name,
            api_key=config.embedding.api_key,
            endpoint=config.embedding.api_endpoint,
            timeout=config.embedding.timeout,
        )

    def add_namespace(self, req: AddNamespaceRequest) -> None:
        err = self.pg_client.add_namespace(req)
        if err is not None:
            raise err

    def add_doc(self, req: AddDocRequest) -> None:
        if not req.vector:
            req.vector = self.emb_client.embedding(req.text)
        err = self.pg_client.add_doc(req)
        if err is not None:
            raise err

    def query(self, req: QueryDocRequest) -> list[DocResponse] | Exception:
        kw_results = self.pg_client.query_text(req)
        if not req.vector:
            req.vector = self.emb_client.embedding(req.query)
        vec_results = self.pg_client.query_vector(req)
        id2doc = {doc.id: doc for doc in kw_results + vec_results}
        return list(id2doc.values())
