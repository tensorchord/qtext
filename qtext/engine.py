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
        self.ranker = config.ranker.ranker(**config.ranker.params)

    def add_namespace(self, req: AddNamespaceRequest) -> None:
        self.pg_client.add_namespace(req)

    def add_doc(self, req: AddDocRequest) -> None:
        if not req.vector:
            req.vector = self.emb_client.embedding(req.text)
        self.pg_client.add_doc(req)

    def query(self, req: QueryDocRequest) -> list[DocResponse]:
        kw_results = self.pg_client.query_text(req)
        if not req.vector:
            req.vector = self.emb_client.embedding(req.query)
        vec_results = self.pg_client.query_vector(req)
        id2doc = {doc.id: doc for doc in kw_results + vec_results}
        ranked = self.ranker.rank(
            req.to_record(),
            [doc.to_record() for doc in id2doc.values()],
        )
        return [DocResponse.from_record(record) for record in ranked]
