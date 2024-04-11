# QText

[![Python Check](https://github.com/tensorchord/qtext/actions/workflows/check.yml/badge.svg)](https://github.com/tensorchord/qtext/actions/workflows/check.yml)
<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/KqswhpVgdU?style=flat"></a>
<a href="https://twitter.com/TensorChord"><img src="https://img.shields.io/twitter/follow/tensorchord?style=social" alt="trackgit-views" /></a>

End-to-end service to query the text with hybrid search and rerank.

Application scenarios:
- Personal knowledge database + search engine
- Rerank experiment and visualization
- RAG pipeline

[![asciicast](https://asciinema.org/a/653540.svg)](https://asciinema.org/a/653540)

## Features

- [x] full text search (Postgres GIN + text search)
- [x] vector similarity search ([pgvecto.rs](https://github.com/tensorchord/pgvecto.rs) HNSW)
- [x] sparse search ([pgvecto.rs](https://github.com/tensorchord/pgvecto.rs) HNSW)
- [x] generate vector and sparse vector if not provided
- [x] reranking
- [x] semantic highlight
- [x] hybrid search explanation
- [x] TUI
- [x] OpenAPI
- [x] OpenMetrics
- [ ] filtering

## How to use

To start all the services with [docker compose](https://docs.docker.com/compose/):

```bash
docker compose -f docker/compose.yaml up -d server
```

Some of the dependent services can be opt-out:
- `emb`: used to generate embedding for query and documents
- `sparse`: used to generate sparse embedding for query and documents (this requires a HuggingFace token that signed the agreement for [prithivida/Splade_PP_en_v1](https://huggingface.co/prithivida/Splade_PP_en_v1))
- `highlight`: used to provide the semantic highlight feature
- `encoder`: rerank with cross-encoder model, you can choose other methods or other online services

For the client example, check:
- [test.py](./test.py): simple demo.
- [test_cohere_wiki.py](./test_cohere_wiki.py): a Wikipedia dataset with Cohere embedding.

## API

We provide a simple sync/async [client](./qtext/client.py). You can also refer to the OpenAPI and build your own client.

- `/api/namespace` POST: create a new namespace and configure the index
- `/api/doc` POST: add a new doc
- `/api/query` POST: query the docs
- `/api/highlight` POST: semantic highlight
- `/metrics` GET: open metrics

Check the [OpenAPI documentation](http://127.0.0.1:8000/openapi/redoc) for more information (this requires the qtext service).

## Terminal UI

We provide a simple terminal UI powered by [Textual](https://github.com/textualize/textual) for you to interact with the service.

```bash
pip install textual
# need to run the qtext service first
python tui/main.py $QTEXT_PORT
```

## Configurations

Check the [config.py](./qtext/config.py) for more detail. It will read the `$HOME/.config/qtext/config.json` if this file exists.

## Integrate to the RAG pipeline

This project has most of the components you need for the RAG except for the last LLM generation step. You can send the retrieval + reranked docs to any LLM providers to get the final result.

## Customize the table schema

> [!NOTE]
> If you already have the table in Postgres, you will be responsible for the text-indexing and vector-indexing part.

1. Define a `dataclass` that includes the **necessary** columns as class attributes
   - annotate the `primary_key`, `text_index`, `vector_index`, `sparse_index` with metadata (not all of them are required, only the necessary ones)
   - attributes without default value or default factory is treated as required when you add new docs
2. Implement the `to_record` and `from_record` methods to be used in the reranking stage
3. Change the `config.vector_store.schema` to the class you have defined

Check the [schema.py](/qtext/schema.py) for more details.
