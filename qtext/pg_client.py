from __future__ import annotations

import numpy as np
import psycopg
from psycopg.adapt import Dumper, Loader
from psycopg.types import TypeInfo

from qtext.log import logger
from qtext.spec import AddDocRequest, AddNamespaceRequest, QueryDocRequest

DISTANCE_TO_METHOD = {
    "euclidean": "vector_l2_ops",
    "cosine": "vector_cos_ops",
    "dot_product": "vector_dot_ops",
}

DISTANCE_TO_OP = {
    "euclidean": "<->",
    "cosine": "<=>",
    "dot_product": "<#>",
}


class VectorDumper(Dumper):
    def dump(self, obj):
        if isinstance(obj, np.ndarray):
            return f"[{','.join(map(str, obj))}]".encode()
        return str(obj).replace(" ", "").encode()


class VectorLoader(Loader):
    def load(self, buf):
        if isinstance(buf, memoryview):
            buf = bytes(buf)
        return np.array(buf.decode()[1:-1].split(","), dtype=np.float32)


async def register_vector_async(conn: psycopg.AsyncConnection):
    info = await TypeInfo.fetch(conn=conn, name="vector")
    register_vector_type(conn, info)


def register_vector(conn: psycopg.Connection):
    info = TypeInfo.fetch(conn=conn, name="vector")
    register_vector_type(conn, info)


def register_vector_type(conn: psycopg.Connection, info: TypeInfo):
    if info is None:
        raise ValueError("vector type not found")
    info.register(conn)

    class VectorTextDumper(VectorDumper):
        oid = info.oid

    adapters = conn.adapters
    adapters.register_dumper(list, VectorTextDumper)
    adapters.register_dumper(np.ndarray, VectorTextDumper)
    adapters.register_loader(info.oid, VectorLoader)


class PgVectorsClient:
    def __init__(self, path: str):
        self.path = path
        self.conn = self.connect()

    def connect(self):
        conn = psycopg.connect(self.path)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vectors;")
        register_vector(conn)
        conn.commit()
        return conn

    def close(self):
        self.conn.close()

    def add_namespace(self, req: AddNamespaceRequest):
        try:
            self.conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {req.name} (
                    id SERIAL PRIMARY KEY,
                    emb vector({req.vector_dim}) NOT NULL,
                    text TEXT NOT NULL,
                    title TEXT,
                    summary TEXT,
                    author TEXT,
                    updated_at TIMESTAMP,
                    tags TEXT[],
                    score REAL DEFAULT 1.0,
                    boost REAL DEFAULT 1.0
                    )
                    """
            )
            self.conn.commit()
            self.conn.execute(
                (
                    f"CREATE INDEX IF NOT EXISTS {req.name}_vectors ON {req.name}"
                    f"USING vectors (emb {DISTANCE_TO_METHOD[req.distance]});"
                )
            )
            self.conn.commit()
        except psycopg.errors.Error as err:
            logger.info("pg client add table error", exc_info=err)
            self.conn.rollback()
            self.conn = self.connect()

    def add_doc(self, req: AddDocRequest):
        try:
            self.conn.execute(
                f"""INSERT INTO {req.namespace} (id, emb, text, title, summary, author, updated_at, tags, score, boost)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                req.doc_id,
                req.vector,
                req.text,
                req.title,
                req.summary,
                req.author,
                req.updated_at,
                req.tags,
                req.score,
                req.boost,
            )
            self.conn.commit()
        except psycopg.errors.Error as err:
            logger.info("pg client add doc error", exc_info=err)
            self.conn.rollback()
            self.conn = self.connect()

    def query_doc(self, req: QueryDocRequest):
        # TODO: get the table index distance
        op = DISTANCE_TO_OP["cosine"]
        try:
            # TODO: filter
            cursor = self.conn.execute(
                f"SELECT *, emb {op} %s AS score FROM {req.namespace} ORDER by score LIMIT %s",
                req.vector,
                req.limit,
            )
        except psycopg.errors.Error as err:
            logger.info("pg client query error", exc_info=err)
            self.conn.rollback()
            self.conn = self.connect()
        return cursor.fetchall()
