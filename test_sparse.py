import httpx

vocab = 30522
dim = 768
namespace = "sparse_test"
client = httpx.Client(base_url="http://127.0.0.1:8000")
resp = client.post(
    "/api/namespace",
    json={"name": namespace, "vector_dim": dim, "sparse_vector_dim": vocab},
)
resp.raise_for_status()

for text in [
    "the early bird, not really catches the worm",
    "Rust is not always faster than Python",
    "Life is short, I use Python",
]:
    resp = client.post(
        "/api/doc",
        json={
            "namespace": namespace,
            "text": text,
        },
    )
    resp.raise_for_status()

resp = client.post(
    "/api/query", json={"namespace": namespace, "query": "Who creates faster Python?"}
)
resp.raise_for_status()
print([(doc["id"], doc["text"]) for doc in resp.json()])
