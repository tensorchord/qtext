from prometheus_client import Counter, Histogram

labels = ("namespace",)

highlight_histogram = Histogram("highlight_latency_seconds", "Highlight cost time")
rerank_histogram = Histogram("rerank_latency_seconds", "ReRank cost time")
doc_counter = Counter("add_doc", "Added documents", labelnames=labels)
embedding_histogram = Histogram("embedding_latency_seconds", "Embedding cost time")
sparse_histogram = Histogram("sparse_latency_seconds", "Sparse embedding cost time")
add_doc_histogram = Histogram(
    "add_doc_latency_seconds", "Add doc cost time", labelnames=labels
)
text_search_histogram = Histogram(
    "full_text_search_latency_seconds",
    "Full text search cost time",
    labelnames=labels,
)
vector_search_histogram = Histogram(
    "vector_search_latency_seconds",
    "Vector search cost time",
    labelnames=labels,
)
sparse_search_histogram = Histogram(
    "sparse_vector_search_latency_seconds",
    "Sparse vector search cost time",
    labelnames=labels,
)
