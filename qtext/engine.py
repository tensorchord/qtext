from __future__ import annotations

from qtext.config import Config
from qtext.emb_client import EmbeddingClient, SparseEmbeddingClient
from qtext.highlight_client import ENGLISH_STOPWORDS, HighlightClient
from qtext.pg_client import PgVectorsClient
from qtext.schema import DefaultTable, Querier
from qtext.spec import (
    AddNamespaceRequest,
    HighlightRequest,
    HighlightResponse,
    QueryDocRequest,
)
from qtext.utils import time_it


class RetrievalEngine:
    def __init__(self, config: Config) -> None:
        self.querier = Querier(config.vector_store.schema)
        self.req_cls = self.querier.generate_request_class()
        self.resp_cls = self.querier.table_type
        self.pg_client = PgVectorsClient(config.vector_store.url, querier=self.querier)
        self.highlight_client = HighlightClient(config.highlight.addr)
        self.emb_client = EmbeddingClient(
            model_name=config.embedding.model_name,
            api_key=config.embedding.api_key,
            endpoint=config.embedding.api_endpoint,
            timeout=config.embedding.timeout,
        )
        self.sparse_client = SparseEmbeddingClient(
            endpoint=config.sparse.addr,
            dim=config.sparse.dim,
            timeout=config.sparse.timeout,
        )
        self.ranker = config.ranker.ranker(**config.ranker.params)

    @time_it
    def add_namespace(self, req: AddNamespaceRequest) -> None:
        self.pg_client.add_namespace(req)

    @time_it
    def add_doc(self, req) -> None:
        if self.querier.has_vector_index():
            vector = self.querier.retrieve_vector(req)
            if not vector and self.querier.has_text_index():
                text = self.querier.retrieve_text(req)
                self.querier.fill_vector(req, self.emb_client.embedding(text=text))
        if self.querier.has_sparse_index():
            sparse = self.querier.retrieve_sparse_vector(req)
            if not sparse and self.querier.has_text_index():
                text = self.querier.retrieve_text(req)
                self.querier.fill_sparse_vector(
                    req, self.sparse_client.sparse_embedding(text=text)
                )
        self.pg_client.add_doc(req)

    @time_it
    def rank(
        self,
        req: QueryDocRequest,
        text_res: list[DefaultTable],
        vector_res: list[DefaultTable],
        sparse_res: list[DefaultTable],
    ) -> list[DefaultTable]:
        docs = self.querier.combine_vector_text(
            vec_res=vector_res, sparse_res=sparse_res, text_res=text_res
        )
        ranked = self.ranker.rank(req.to_record(), docs)
        return [DefaultTable.from_record(record) for record in ranked]

    @time_it
    def query(self, req: QueryDocRequest) -> list[DefaultTable]:
        kw_results = self.pg_client.query_text(req)
        if self.querier.has_vector_index() and not req.vector:
            req.vector = self.emb_client.embedding(req.query)
        if self.querier.has_sparse_index() and not req.sparse_vector:
            req.sparse_vector = self.sparse_client.sparse_embedding(req.query)
        vec_results = self.pg_client.query_vector(req)
        sparse_results = self.pg_client.query_sparse_vector(req)
        return self.rank(req, kw_results, vec_results, sparse_results)

    @time_it
    def highlight(self, req: HighlightRequest) -> HighlightResponse:
        text_scores = self.highlight_client.highlight_score(req.query, req.docs)
        highlighted = []
        for text_score in text_scores:
            words = []
            highlight_index = set()
            index = -1
            for word in text_score:
                if word.text.startswith("##"):
                    words[-1] += word.text[2:]
                    if word.score >= req.threshold:
                        highlight_index.add(index)
                    continue

                words.append(word.text)
                index += 1
                if req.ignore_stopwords and word.text.lower() in ENGLISH_STOPWORDS:
                    continue
                if word.score >= req.threshold:
                    highlight_index.add(index)

            highlighted.append(
                " ".join(
                    word if i not in highlight_index else req.template.format(word)
                    for i, word in enumerate(words)
                )
            )
        return HighlightResponse(highlighted=highlighted)
