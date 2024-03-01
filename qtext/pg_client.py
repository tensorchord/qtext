from __future__ import annotations

import numpy as np
import psycopg
from psycopg.adapt import Dumper, Loader
from psycopg.types import TypeInfo

from qtext.log import logger
from qtext.spec import (
    DISTANCE_TO_METHOD,
    DISTANCE_TO_OP,
    AddDocRequest,
    AddNamespaceRequest,
    DocResponse,
    QueryDocRequest,
)


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
            self.conn.execute(
                f"ALTER TABLE {req.name} ADD COLUMN fts_vector tsvector GENERATED "
                "ALWAYS AS (to_tsvector('english', text)) stored;"
            )
            self.conn.execute(
                (
                    f"CREATE INDEX IF NOT EXISTS {req.name}_vectors ON {req.name} "
                    f"USING vectors (emb {DISTANCE_TO_METHOD[req.distance]});"
                )
            )
            self.conn.commit()
        except psycopg.errors.Error as err:
            logger.info("pg client add table error", exc_info=err)
            self.conn.rollback()
            raise RuntimeError("add namespace error") from err

    def add_doc(self, req: AddDocRequest):
        try:
            attributes = [
                "id",
                "emb",
                "text",
                "title",
                "summary",
                "author",
                "updated_at",
                "tags",
                "score",
                "boost",
            ]
            placeholders = [
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
            ]
            if req.doc_id is None:
                attributes.pop(0)
                placeholders.pop(0)
            self.conn.execute(
                (
                    f"INSERT INTO {req.namespace} ({','.join(attributes)})"
                    f"VALUES ({','.join(['%s']*len(placeholders))})"
                ),
                placeholders,
            )
            self.conn.commit()
        except psycopg.errors.Error as err:
            logger.info("pg client add doc error", exc_info=err)
            self.conn.rollback()
            raise RuntimeError("add doc error") from err

    def query_text(self, req: QueryDocRequest) -> list[DocResponse]:
        try:
            cursor = self.conn.execute(
                (
                    "SELECT id, text, emb, score, boost, title, summary, author,"
                    "updated_at, tags, ts_rank_cd(fts_vector, query) AS rank "
                    f"FROM {req.namespace}, to_tsquery(%s) query "
                    "WHERE query @@ fts_vector order by rank desc LIMIT %s"
                ),
                (" | ".join(req.query.strip().split(" ")), req.limit),
            )
        except psycopg.errors.Error as err:
            logger.info("pg client query error", exc_info=err)
            self.conn.rollback()
            raise RuntimeError("query text error") from err
        return [DocResponse.from_query_result(res) for res in cursor.fetchall()]

    def query_vector(self, req: QueryDocRequest) -> list[DocResponse]:
        op = DISTANCE_TO_OP[req.distance]
        try:
            # TODO: filter
            cursor = self.conn.execute(
                (
                    "SELECT id, text, emb, score, boost, title, summary, author,"
                    f"updated_at, tags, emb {op} %s AS distance "
                    f"FROM {req.namespace} ORDER by distance LIMIT %s"
                ),
                (req.vector, req.limit),
            )
        except psycopg.errors.Error as err:
            logger.info("pg client query error", exc_info=err)
            self.conn.rollback()
            raise RuntimeError("query vector error") from err
        return [DocResponse.from_query_result(res) for res in cursor.fetchall()]
