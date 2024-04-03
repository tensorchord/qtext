from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Type

import msgspec
from msgspec import NODEFAULT
from msgspec.inspect import (
    DateTimeType,
    DictType,
    Field,
    FloatType,
    IntType,
    ListType,
    NoneType,
    StrType,
    UnionType,
)

from qtext.spec import Record


@dataclass(kw_only=True)
class DefaultTable:
    id: int | None = field(default=None, metadata={"primary_key": True})
    text: str = field(metadata={"text_index": True})
    vector: list[float] = field(default_factory=list, metadata={"vector_index": True})
    sparse_vector: list[float] = field(
        default_factory=list, metadata={"sparse_index": True}
    )
    title: str | None = field(default=None, metadata={"text_index": True})
    summary: str | None = None
    author: str | None = None
    updated_at: datetime | None = None
    tags: list[str] | None = None
    score: float = 1.0
    boost: float = 1.0

    def to_record(self) -> Record:
        return Record(
            id=self.id,
            text=self.text,
            vector=self.vector,
            sparse_vector=self.sparse_vector,
            score=self.score,
            boost=self.boost,
            title=self.title,
            summary=self.summary,
            author=self.author,
            updated_at=self.updated_at,
            tags=self.tags,
        )

    @classmethod
    def from_record(cls, record: Record) -> DefaultTable:
        """Convert from a Record to a DefaultTable instance.

        This method affects the final output of the query. Some of the fields can be
        omitted before returning to the user to avoid large transfers of data.
        """
        return cls(
            id=record.id,
            text=record.text,
            score=record.score,
            boost=record.boost,
            title=record.title,
            summary=record.summary,
            author=record.author,
            updated_at=record.updated_at,
            tags=record.tags,
        )


class Querier:
    def __init__(self, table: Type[DefaultTable]) -> None:
        self.table_type = table
        self.fields: list[Field] = msgspec.inspect.type_info(table).fields
        self.primary_key: str | None = None
        self.vector_column: str | None = None
        self.sparse_column: str | None = None
        self.text_columns: list[str] = []

        for f in fields(self.table_type):
            if f.metadata.get("primary_key"):
                self.primary_key = f.name
            if f.metadata.get("vector_index"):
                self.vector_column = f.name
            if f.metadata.get("sparse_index"):
                self.sparse_column = f.name
            if f.metadata.get("text_index"):
                self.text_columns.append(f.name)

    def generate_request_class(self) -> DefaultTable:
        """Generate the user request class."""

        @dataclass(kw_only=True)
        class Request(self.table_type):
            namespace: str

        return Request

    def generate_response_class(self) -> DefaultTable:
        """Generate the class used by the raw dict data from postgres."""

        @dataclass(kw_only=True)
        class Response(self.table_type):
            rank: float

        return Response

    def fill_vector(self, obj, vector: list[float]):
        setattr(obj, self.vector_column, vector)

    def retrieve_vector(self, obj):
        return getattr(obj, self.vector_column)

    def fill_sparse_vector(self, obj, vector: list[float]):
        setattr(obj, self.sparse_column, vector)

    def retrieve_sparse_vector(self, obj):
        return getattr(obj, self.sparse_column)

    def retrieve_text(self, obj):
        return "\n".join(getattr(obj, t, "") or "" for t in self.text_columns)

    def combine_vector_text(
        self,
        vec_res: list[DefaultTable],
        sparse_res: list[DefaultTable],
        text_res: list[DefaultTable],
    ) -> list[Record]:
        """Combine hybrid search results."""
        id_to_record = {}
        for vec in vec_res:
            record = vec.to_record()
            record.vector_sim = vec.rank
            id_to_record[record.id] = record

        for sparse in sparse_res:
            record = sparse.to_record()
            if record.id not in id_to_record:
                id_to_record[record.id] = record
            id_to_record[record.id].title_sim = sparse.rank

        for text in text_res:
            record = text.to_record()
            if record.id not in id_to_record:
                id_to_record[record.id] = record
            id_to_record[record.id].content_bm25 = text.rank

        return list(id_to_record.values())

    @staticmethod
    def to_pg_type(field_type: msgspec.inspect.Type) -> str:
        if isinstance(field_type, UnionType):
            for t in field_type.types:
                if isinstance(t, NoneType):
                    # ignore None
                    continue
                return Querier.to_pg_type(t)

        if isinstance(field_type, ListType):
            return Querier.to_pg_type(field_type.item_type) + "[]"

        return {
            StrType: "TEXT",
            IntType: "INTEGER",
            FloatType: "REAL",
            DateTimeType: "TIMESTAMP",
            DictType: "JSONB",
        }[field_type.__class__]

    def create_table(self, name: str, dim: int, sparse_dim: int) -> str:
        # check the vector dimension provided
        if self.has_vector_index() and dim == 0:
            raise ValueError(
                "Vector dimension is required when schema has vector index"
            )
        if self.has_sparse_index() and sparse_dim == 0:
            raise ValueError(
                "Sparse vector dimension is required when schema has sparse index"
            )

        sql = f"CREATE TABLE IF NOT EXISTS {name} ("
        for i, f in enumerate(self.fields):
            if f.name == self.primary_key:
                sql += f"{f.name} SERIAL PRIMARY KEY, "
                continue
            elif f.name == self.vector_column:
                sql += f"{f.name} vector({dim}) "
            elif f.name == self.sparse_column:
                sql += f"{f.name} svector({sparse_dim}) "
            else:
                sql += f"{f.name} {Querier.to_pg_type(f.type)} "

            if f.required:
                sql += "NOT NULL "
            if f.default not in (NODEFAULT, None):
                sql += f"DEFAULT {f.default} "

            if i < len(self.fields) - 1:
                sql += ", "
        return sql + ");"

    def has_vector_index(self) -> bool:
        return self.vector_column is not None

    def has_sparse_index(self) -> bool:
        return self.sparse_column is not None

    def has_text_index(self) -> bool:
        return len(self.text_columns) > 0

    def vector_index(self, table: str) -> str:
        """
        This assumes that all the vectors are normalized, so inner product
        is used since it can be computed efficiently.
        """
        if not self.has_vector_index():
            return ""
        return (
            f"CREATE INDEX IF NOT EXISTS {table}_vectors ON {table} USING "
            f"vectors ({self.vector_column} vector_dot_ops);"
        )

    def sparse_index(self, table: str) -> str:
        if not self.has_sparse_index():
            return ""
        return (
            f"CREATE INDEX IF NOT EXISTS {table}_sparse ON {table} USING "
            f"vectors ({self.sparse_column} svector_dot_ops);"
        )

    def text_index(self, table: str) -> str:
        """
        refer to https://dba.stackexchange.com/a/164081
        """
        if not self.has_text_index():
            return ""
        indexed_columns = (
            self.text_columns[0]
            if len(self.text_columns) == 1
            else f"immutable_concat_ws('. ', {', '.join(self.text_columns)})"
        )
        return (
            "CREATE OR REPLACE FUNCTION immutable_concat_ws(text, VARIADIC text[]) "
            "RETURNS text LANGUAGE sql IMMUTABLE PARALLEL SAFE "
            "RETURN array_to_string($2, $1);"
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS fts_vector tsvector "
            f"GENERATED ALWAYS AS (to_tsvector('english', {indexed_columns})) stored; "
            f"CREATE INDEX IF NOT EXISTS ts_idx ON {table} USING GIN (fts_vector);"
        )

    def vector_query(self, table: str) -> str:
        columns = ", ".join(f.name for f in self.fields)
        return (
            f"SELECT {columns}, {self.vector_column} <#> %s AS rank "
            f"FROM {table} ORDER by rank LIMIT %s;"
        )

    def sparse_query(self, table: str) -> str:
        columns = ", ".join(f.name for f in self.fields)
        return (
            f"SELECT {columns}, {self.sparse_column} <#> %s AS rank "
            f"FROM {table} ORDER by rank LIMIT %s;"
        )

    def text_query(self, table: str) -> str:
        columns = ", ".join(f.name for f in self.fields)
        return (
            f"SELECT {columns}, ts_rank_cd(fts_vector, query) AS rank "
            f"FROM {table}, to_tsquery(%s) query "
            "WHERE fts_vector @@ query order by rank desc LIMIT %s;"
        )

    def columns(self) -> list[str]:
        return list(f.name for f in self.fields)


if __name__ == "__main__":
    search = Querier(DefaultTable)
    print(search.create_table("document", 64))
    print(search.vector_index("document"))
    print(search.text_index("document"))
    print(search.vector_query("document"))
    print(search.text_query("document"))

    req_cls = search.generate_request_class()
    resp_cls = search.generate_response_class()
