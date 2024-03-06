from os import environ

import cohere
import httpx
from datasets import load_dataset
from tqdm import tqdm

namespace = "cohere_wiki"
dim = 768
client = httpx.Client(base_url="http://127.0.0.1:8000")
resp = client.post("/api/namespace", json={"name": namespace, "vector_dim": dim})
resp.raise_for_status()

docs = load_dataset(
    "Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True
)

count = 0
for doc in tqdm(docs):
    resp = client.post(
        "/api/doc",
        json={
            "namespace": namespace,
            "text": doc["text"],
            "doc_id": doc["id"],
            "vector": doc["emb"],
            "title": doc["title"],
        },
    )
    if resp.is_error:
        print(f"Error adding doc: ({resp.status_code}) {resp.text}")
        continue
    count += 1

print(f"Added {count} docs")

query = "the cat is on the mat"
co = cohere.Client(api_key=environ["COHERE_TOKEN"])
emb_resp = co.embed([query], model="multilingual-22-12")
resp = client.post(
    "/api/query",
    json={
        "namespace": namespace,
        "query": query,
        "vector": emb_resp.embeddings[0],
    },
)
resp.raise_for_status()
for doc in resp.json():
    print(f"[{doc['id']}] {doc['title']}")
    print(doc["text"])
    print("=" * 80)
