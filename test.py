from qtext.spec import AddNamespaceRequest, AddDocRequest, QueryDocRequest
from qtext.pg_client import PgVectorsClient


client = PgVectorsClient("postgresql://postgres:password@127.0.0.1:5432/")
# client.add_namespace(AddNamespaceRequest(name="document", vector_dim=5))
client.add_doc(AddDocRequest(namespace="document", text="the early bird, not really catches the worm", title="test, punctuation", vector=[0.4, 0.3, 0.1]))

print(client.query_text(QueryDocRequest(namespace="document", query="Rust is faster than Python")))
print(client.query_vector(QueryDocRequest(namespace="document", vector=[0.4, 0.3, 0.1], query="")))