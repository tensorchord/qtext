from __future__ import annotations

import msgspec
import numpy as np
import onnxruntime as ort
from mosec import Server, Worker
from transformers import AutoTokenizer

MODEL_NAME = "vespa-engine/col-minilm"


class Token(msgspec.Struct, kw_only=True):
    text: str
    id: int
    vector: np.ndarray


class HighlightToken(msgspec.Struct, kw_only=True):
    text: str
    score: float


class Highlight(Worker):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.session = ort.InferenceSession("model_quantized.onnx")

    def forward(self, queries: list[str]) -> list[list[HighlightToken]]:
        """
        Args:
            queries: 1st is the query, the rest are documents
        Returns:
            the max similarity for each token in the documents
        """
        tokens = self.tokenizer(queries, padding=True, return_tensors="np")
        outputs = self.session.run(
            ["contextual"],
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            },
        )[0]
        token_vectors = []
        for ids, masks, vectors in zip(
            tokens["input_ids"], tokens["attention_mask"], outputs
        ):
            token_vector = []
            for id, mask, vector in zip(ids, masks, vectors):
                if id in self.tokenizer.all_special_ids or mask == 0:
                    continue
                token_vector.append(
                    Token(text=self.tokenizer.decode(id), id=id, vector=vector)
                )
            token_vectors.append(token_vector)

        similarities = []
        for i in range(1, len(queries)):
            similarities.append(
                [
                    HighlightToken(
                        score=max(
                            token.vector @ query_token.vector
                            for query_token in token_vectors[0]
                        ).tolist(),
                        text=token.text,
                    )
                    for token in token_vectors[i]
                ]
            )

        return similarities

    def serialize(self, obj):
        return msgspec.json.encode(obj)


if __name__ == "__main__":
    server = Server()
    server.append_worker(Highlight, num=1)
    server.run()
