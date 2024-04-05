from __future__ import annotations

from time import perf_counter

import numpy as np
import psycopg
from psycopg import sql
from psycopg.adapt import Dumper, Loader
from psycopg.rows import dict_row
from psycopg.types import TypeInfo

from qtext.log import logger
from qtext.metrics import (
    doc_counter,
    sparse_search_histogram,
    text_search_histogram,
    vector_search_histogram,
)
from qtext.schema import DefaultTable, Querier
from qtext.spec import AddNamespaceRequest, QueryDocRequest, SparseEmbedding
from qtext.utils import time_it


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


class SparseVectorDumper(Dumper):
    def dump(self, obj):
        if isinstance(obj, np.ndarray):
            return f"[{','.join(map(str, obj))}]".encode()
        if isinstance(obj, SparseEmbedding):
            return obj.to_str().encode()
        raise ValueError(f"unsupported type {type(obj)}")


def register_sparse_vector(conn: psycopg.Connection):
    info = TypeInfo.fetch(conn=conn, name="svector")
    register_svector_type(conn, info)


def register_svector_type(conn: psycopg.Connection, info: TypeInfo):
    if info is None:
        raise ValueError("vector type not found")
    info.register(conn)

    class SparseVectorTextDumper(SparseVectorDumper):
        oid = info.oid

    adapters = conn.adapters
    adapters.register_dumper(SparseEmbedding, SparseVectorTextDumper)
    adapters.register_dumper(np.ndarray, SparseVectorTextDumper)
    adapters.register_loader(info.oid, VectorLoader)


class PgVectorsClient:
    def __init__(self, path: str, querier: Querier):
        self.path = path
        self.querier = querier
        self.resp_cls = self.querier.generate_response_class()
        self.conn = self.connect()

    def connect(self):
        conn = psycopg.connect(self.path, row_factory=dict_row)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vectors;")
        register_vector(conn)
        register_sparse_vector(conn)
        conn.commit()
        return conn

    def close(self):
        self.conn.close()

    @time_it
    def add_namespace(self, req: AddNamespaceRequest):
        try:
            create_table_sql = self.querier.create_table(
                req.name, req.vector_dim, req.sparse_vector_dim
            )
            vector_index_sql = self.querier.vector_index(req.name)
            sparse_index_sql = self.querier.sparse_index(req.name)
            text_index_sql = self.querier.text_index(req.name)
            self.conn.execute(create_table_sql)
            self.conn.execute(vector_index_sql)
            self.conn.execute(sparse_index_sql)
            self.conn.execute(text_index_sql)
            self.conn.commit()
        except psycopg.errors.Error as err:
            logger.info("pg client create table error", exc_info=err)
            self.conn.rollback()
            raise RuntimeError("add namespace error") from err

    def add_doc(self, req):
        try:
            attributes = self.querier.columns()
            primary_id = self.querier.primary_key
            if primary_id is None or getattr(req, primary_id, None) is None:
                attributes.remove(primary_id)
            placeholders = [getattr(req, key) for key in attributes]
            self.conn.execute(
                sql.SQL(
                    "INSERT INTO {table} ({fields}) VALUES ({placeholders})"
                ).format(
                    table=sql.Identifier(req.namespace),
                    fields=sql.SQL(",").join(map(sql.Identifier, attributes)),
                    placeholders=sql.SQL(",").join(
                        sql.Placeholder() for _ in range(len(placeholders))
                    ),
                ),
                placeholders,
            )
            self.conn.commit()
            doc_counter.labels(req.namespace).inc()
        except psycopg.errors.Error as err:
            logger.info("pg client add doc error", exc_info=err)
            self.conn.rollback()
            raise RuntimeError("add doc error") from err

    @time_it
    def query_text(self, req: QueryDocRequest) -> list[DefaultTable]:
        if not self.querier.has_text_index():
            logger.debug("skip text query since there is no text index")
            return []
        try:
            start_time = perf_counter()
            cursor = self.conn.execute(
                self.querier.text_query(req.namespace),
                (" | ".join(req.query.strip().split(" ")), req.limit),
            )
            results = cursor.fetchall()
            text_search_histogram.labels(req.namespace).observe(
                perf_counter() - start_time
            )
        except psycopg.errors.Error as err:
            logger.info("pg client query text error", exc_info=err)
            self.conn.rollback()
            raise RuntimeError("query text error") from err
        return [self.resp_cls(**res) for res in results]

    @time_it
    def query_vector(self, req: QueryDocRequest) -> list[DefaultTable]:
        if not self.querier.has_vector_index():
            logger.debug("skip vector query since there is no vector index")
            return []
        try:
            # TODO: filter
            start_time = perf_counter()
            cursor = self.conn.execute(
                self.querier.vector_query(req.namespace),
                (req.vector, req.limit),
            )
            results = cursor.fetchall()
            vector_search_histogram.labels(req.namespace).observe(
                perf_counter() - start_time
            )
        except psycopg.errors.Error as err:
            logger.info("pg client query vector error", exc_info=err)
            self.conn.rollback()
            raise RuntimeError("query vector error") from err
        return [self.resp_cls(**res) for res in results]

    @time_it
    def query_sparse_vector(self, req: QueryDocRequest) -> list[DefaultTable]:
        if not self.querier.has_sparse_index():
            logger.debug("skip sparse vector query since there is no sparse index")
            return []
        try:
            start_time = perf_counter()
            cursor = self.conn.execute(
                self.querier.sparse_query(req.namespace),
                (req.sparse_vector, req.limit),
            )
            results = cursor.fetchall()
            sparse_search_histogram.labels(req.namespace).observe(
                perf_counter() - start_time
            )
        except psycopg.errors.Error as err:
            logger.info("pg client query sparse vector error", exc_info=err)
            self.conn.rollback()
            raise RuntimeError("query sparse vector error") from err
        return [self.resp_cls(**res) for res in results]
