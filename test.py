from datetime import datetime, timedelta

import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000")
resp = client.post("/api/namespace", json={"name": "document", "vector_dim": 768})
resp.raise_for_status()
for i, text in enumerate([
    "the early bird, not really catches the worm",
    "Rust is not always faster than Python",
    "Life is short, I use Python",
]):
    resp = client.post(
        "/api/doc",
        json={
            "namespace": "document",
            "text": text,
            "updated_at": str(datetime.now() - timedelta(days=i)),
        },
    )
    resp.raise_for_status()

resp = client.post(
    "/api/query", json={"namespace": "document", "query": "Who creates Python?"}
)
resp.raise_for_status()
print([(doc["id"], doc["text"]) for doc in resp.json()])
resp = client.post(
    "/api/highlight", json={"query": "Python", "docs": ["Life is short, I use Python"]}
)
resp.raise_for_status()
print(resp.json())
