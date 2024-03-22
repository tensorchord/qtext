# QText

End-to-end service to query the text.

- [x] full text search (Postgres GIN + text search)
- [x] vector similarity search ([pgvecto.rs](https://github.com/tensorchord/pgvecto.rs) HNSW)
- [x] generate vector if not provided
- [ ] sparse search
- [ ] filtering
- [x] reranking with [reranker](https://github.com/kemingy/reranker)
- [x] semantic highlight

## How to use

To start all the services:

```bash
docker compose -f docker/compose.yaml up -d server
```

Some of the dependent services can be opt-out:
- `emb`: used to generate embedding for query and documents
- `colbert`: used to provide the semantic highlight feature
- `encoder`: rerank with cross-encoder model, you can choose other methods or other online services

For the client example, check:
- [test.py](./test.py): simple demo.
- [test_cohere_wiki.py](./test_cohere_wiki.py): if you have the Cohere Token. Remember to change the `config.ranker.ranker` to the `CohereClient` (imported from `reranker`).

## API

- `/api/namespace` POST: create a new namespace and configure the text + vector index
- `/api/doc` POST: add a new doc
- `/api/query` POST: query the docs
- `/api/highlight` POST: semantic highlight

Check the [OpenAPI documentation](http://127.0.0.1:8000/openapi/redoc) for more information (this requires the qtext service).

## Configurations

Check the [config.py](./qtext/config.py) for more detail. It will read the `$HOME/.config/qtext/config.json` if this file exists.

## Integrate to the RAG pipeline

This project has most of the components you need for the RAG except for the last LLM generation step. You can send the retrieval + reranked docs to any LLM providers to get the final result.

## Customize the table schema

> [!NOTE]
> If you already have the table in Postgres, you will be responsible for the text-indexing and vector-indexing part.

1. Define a `dataclass` that includes the **necessary** columns as class attributes
   - annotate the `primary_key`, `text_index`, `vector_index` with metadata
   - attributes without default value or default factory is treated as required when you add new docs
2. Implement the `to_record` and `from_record` methods to be used in the reranking stage
3. Change the `config.vector_store.schema` to the class you have defined

Check the [schema.py](/qtext/schema.py) for more details.
