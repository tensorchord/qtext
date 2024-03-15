# QText

End-to-end service to query the text.

## How to use

```bash
docker compose -f docker/compose.yaml up -d server
```

Check the [OpenAPI documentation](http://127.0.0.1:8000/openapi/redoc) for more information.

## Customize the table schema

1. Define a dataclass that includes all the columns as class attributes
2. Implement the `to_record` and `from_record` methods to be used in the reranking

Check the [schema.py](/qtext/schema.py) for more details.
