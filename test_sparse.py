import numpy as np
import psycopg
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
from psycopg.adapt import Dumper, Loader
from psycopg.rows import dict_row
from psycopg.types import TypeInfo

model = SparseTextEmbedding("prithvida/Splade_PP_en_v1")

vocab = 30522
vec = next(model.embed("the quick brown fox jumped over the lazy dog"))
print(vec.values.shape, vec.indices.shape)


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


class SVectorDumper(Dumper):
    def dump(self, obj):
        if isinstance(obj, np.ndarray):
            return f"[{','.join(map(str, obj))}]".encode()
        return str(obj).replace(" ", "").encode()


class SVectorLoader(Loader):
    def load(self, buf):
        if isinstance(buf, memoryview):
            buf = bytes(buf)
        return np.array(buf.decode()[1:-1].split(","), dtype=np.float32)


def register_vector(conn: psycopg.Connection):
    info = TypeInfo.fetch(conn=conn, name="vector")
    register_vector_type(conn, info)
    sinfo = TypeInfo.fetch(conn=conn, name="svector")
    register_svector_type(conn, sinfo)


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


def register_svector_type(conn: psycopg.Connection, info: TypeInfo):
    if info is None:
        raise ValueError("vector type not found")
    info.register(conn)

    class SVectorTextDumper(SVectorDumper):
        oid = info.oid

    adapters = conn.adapters
    adapters.register_dumper(list, SVectorTextDumper)
    adapters.register_dumper(np.ndarray, SVectorTextDumper)
    adapters.register_loader(info.oid, SVectorLoader)


vocab = 30522

conn = psycopg.connect(
    "postgresql://postgres:password@127.0.0.1:5432/",
    autocommit=True,
    row_factory=dict_row,
)
register_vector(conn)
conn.execute("create extension if not exists vectors;")
conn.execute(
    f"create table if not exists sparse (id serial primary key, vec svector({vocab}), text text);"
)

indices = np.array([10, 233])
values = np.array([0.23, 0.11])
z = np.zeros(30522)
z[indices] = values

conn.execute(
    "insert into sparse (vec, text) values (%s, %s)", (z, "hello there"), binary=True
)

cur = conn.execute("select * from sparse;")
for row in cur.fetchall():
    print(row["id"], row["text"])
