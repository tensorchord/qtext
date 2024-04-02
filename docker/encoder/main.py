from os import environ

from mosec import Server, Worker
from mosec.mixin import TypedMsgPackMixin
from msgspec import Struct
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
WORKER_NUM = environ.get("WORKER_NUM", 1)


class Request(Struct, kw_only=True):
    query: str
    docs: list[str]


class Response(Struct, kw_only=True):
    scores: list[float]


class Encoder(TypedMsgPackMixin, Worker):
    def __init__(self):
        self.model_name = environ.get("MODEL_NAME", DEFAULT_MODEL)
        self.model = CrossEncoder(self.model_name)

    def forward(self, req: Request) -> Response:
        scores = self.model.predict([[req.query, doc] for doc in req.docs])
        return Response(scores=scores.tolist())


if __name__ == "__main__":
    server = Server()
    server.append_worker(Encoder, num=WORKER_NUM)
    server.run()
