from __future__ import annotations

from qtext.config import Config
from qtext.emb_client import EmbeddingClient
from qtext.highlight_client import ENGLISH_STOPWORDS, HighlightClient
from qtext.pg_client import PgVectorsClient
from qtext.schema import DefaultTable, Querier
from qtext.spec import (
    AddDocRequest,
    AddNamespaceRequest,
    HighlightRequest,
    HighlightResponse,
    QueryDocRequest,
)
from qtext.utils import time_it


class RetrievalEngine:
    def __init__(self, config: Config) -> None:
        self.querier = Querier(config.vector_store.schema)
        self.pg_client = PgVectorsClient(config.vector_store.url, querier=self.querier)
        self.highlight_client = HighlightClient(config.highlight.addr)
        self.emb_client = EmbeddingClient(
            model_name=config.embedding.model_name,
            api_key=config.embedding.api_key,
            endpoint=config.embedding.api_endpoint,
            timeout=config.embedding.timeout,
        )
        self.ranker = config.ranker.ranker(**config.ranker.params)

    @time_it
    def add_namespace(self, req: AddNamespaceRequest) -> None:
        self.pg_client.add_namespace(req)

    @time_it
    def add_doc(self, req: AddDocRequest) -> None:
        if not req.vector:
            req.vector = self.emb_client.embedding(req.text)
        self.pg_client.add_doc(req)

    @time_it
    def query(self, req: QueryDocRequest) -> list[DefaultTable]:
        kw_results = self.pg_client.query_text(req)
        if not req.vector:
            req.vector = self.emb_client.embedding(req.query)
        vec_results = self.pg_client.query_vector(req)
        records = self.querier.combine_vector_text(vec_results, kw_results)
        ranked = self.ranker.rank(req.to_record(), records)
        return [DefaultTable.from_record(record) for record in ranked]

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
