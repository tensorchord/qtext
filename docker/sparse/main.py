from __future__ import annotations

import msgspec
import numpy as np
import onnxruntime as ort
from mosec import Server, Worker
from transformers import AutoTokenizer

MODEL_NAME = "prithivida/Splade_PP_en_v1"


class SparseEmbedding(msgspec.Struct, kw_only=True, frozen=True):
    dim: int
    indices: list[int]
    values: list[float]


class SpladePP(Worker):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.session = ort.InferenceSession("model.onnx")
        self.dim = self.tokenizer.vocab_size

    def forward(self, queries: list[str]) -> list[SparseEmbedding]:
        tokens = self.tokenizer(queries, padding=True, return_tensors="np")
        outputs = self.session.run(
            None,
            {
                "input_ids": tokens["input_ids"],
                "input_mask": tokens["attention_mask"],
                "segment_ids": tokens["token_type_ids"],
            },
        )[0]

        relu_log = np.log(1 + np.maximum(outputs, 0))
        weighted_log = relu_log * np.expand_dims(tokens["attention_mask"], axis=-1)
        scores = np.max(weighted_log, axis=1)

        results = []
        for row in scores:
            indices = row.nonzero()[0]
            values = row[indices]
            results.append(
                SparseEmbedding(
                    dim=self.dim, indices=indices.tolist(), values=values.tolist()
                )
            )
        return results

    def serialize(self, obj):
        return msgspec.json.encode(obj)


if __name__ == "__main__":
    server = Server()
    server.append_worker(SpladePP, num=1)
    server.run()
