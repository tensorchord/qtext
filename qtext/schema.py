from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Type

import msgspec
from msgspec import NODEFAULT
from msgspec.inspect import (
    DateTimeType,
    Field,
    FloatType,
    IntType,
    ListType,
    NoneType,
    StrType,
    UnionType,
)
from reranker import Record


@dataclass
class DefaultTable:
    id: int = field(metadata={"primary_key": True})
    text: str = field(metadata={"text_index": True})
    vector: list[float] = field(metadata={"vector_index": True})
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
        return cls(
            id=record.id,
            text=record.text,
            vector=record.vector,
            score=record.score,
            boost=record.boost,
            title=record.title,
            summary=record.summary,
            author=record.author,
            updated_at=record.updated_at,
            tags=record.tags,
        )


class SearchTable:
    def __init__(self, table: Type[DefaultTable]) -> None:
        self.table_type = table
        self.fields: list[Field] = msgspec.inspect.type_info(table).fields
        self.primary_key: str | None = None
        self.vector_column: str | None = None
        self.text_columns: list[str] = []

        for f in table.__dataclass_fields__.values():
            if f.metadata.get("primary_key"):
                self.primary_key = f.name
            if f.metadata.get("vector_index"):
                self.vector_column = f.name
            if f.metadata.get("text_index"):
                self.text_columns.append(f.name)

    @staticmethod
    def to_pg_type(field_type: msgspec.inspect.Type) -> str:
        if isinstance(field_type, UnionType):
            for t in field_type.types:
                if isinstance(t, NoneType):
                    # ignore None
                    continue
                return SearchTable.to_pg_type(t)

        if isinstance(field_type, ListType):
            return SearchTable.to_pg_type(field_type.item_type) + "[]"

        return {
            StrType: "TEXT",
            IntType: "INTEGER",
            FloatType: "REAL",
            DateTimeType: "TIMESTAMP",
        }[field_type.__class__]

    def create_table(self, name: str, dim: int) -> str:
        sql = f"CREATE TABLE IF NOT EXISTS {name} ("
        for i, f in enumerate(self.fields):
            if f.name == self.primary_key:
                sql += f"{f.name} SERIAL PRIMARY KEY, "
                continue
            elif f.name == self.vector_column:
                sql += f"{f.name} vector({dim}) "
            else:
                sql += f"{f.name} {SearchTable.to_pg_type(f.type)} "

            if f.required:
                sql += "NOT NULL "
            if f.default not in (NODEFAULT, None):
                sql += f"DEFAULT {f.default} "

            if i < len(self.fields) - 1:
                sql += ", "
        return sql + ")"

    def vector_index(self, table: str) -> str:
        return (
            f"CREATE INDEX IF NOT EXISTS {table}_vectors ON {table} USING "
            f"vectors ({self.vector_column} vector_dot_ops);"
        )

    def text_index(self, table: str) -> str:
        indexed_columns = " || ' ' || ".join(self.text_columns)
        return (
            f"ALTER TABLE {table} ADD COLUMN fts_vector tsvector GENERATED "
            f"ALWAYS AS (to_tsvector('english', {indexed_columns})) stored;"
        )


if __name__ == "__main__":
    search = SearchTable(DefaultTable)
    print(search.create_table("document", 64))
    print(search.vector_index("document"))
    print(search.text_index("document"))
