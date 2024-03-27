from datetime import datetime, timedelta

import httpx

namespace = "document"
dim = 768

client = httpx.Client(base_url="http://127.0.0.1:8000")
resp = client.post("/api/namespace", json={"name": namespace, "vector_dim": dim})
resp.raise_for_status()
for i, text in enumerate(
    [
        "the early bird, not really catches the worm",
        "Rust is not always faster than Python",
        "Life is short, I use Python",
    ]
):
    resp = client.post(
        "/api/doc",
        json={
            "namespace": namespace,
            "text": text,
            "updated_at": str(datetime.now() - timedelta(days=i)),
        },
    )
    resp.raise_for_status()

resp = client.post(
    "/api/query", json={"namespace": namespace, "query": "Who creates faster Python?"}
)
resp.raise_for_status()
print([(doc["id"], doc["text"]) for doc in resp.json()])
resp = client.post(
    "/api/highlight",
    json={
        "query": "Effects of climate change on marine ecosystems",
        "docs": [
            "The changing climate has profound impacts on marine ecosystems.",
            "Rising temperatures, ocean acidification, and altered precipitation patterns all contribute to shifts in the distribution and behavior of marine species, influencing the delicate balance of underwater ecosystems.",
        ],
    },
)
resp.raise_for_status()
print(resp.json())
