from __future__ import annotations

import sys
from typing import ClassVar

from httpx import HTTPStatusError
from textual.app import App, ComposeResult, on
from textual.containers import Container, HorizontalScroll, ScrollableContainer
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from qtext.client import QTextAsyncClient
from qtext.spec import RankedResponse, RetrieveResponse

PORT = 8000
RANKED_COLUMNS = (
    "ranked_idx",
    "vector_idx",
    "sparse_idx",
    "fts_idx",
    "id",
    "title",
    "text",
)
RETRIEVE_COLUMNS = ("id", "title", "text")


class Form(Container):
    BORDER_TITLE = "Input the query here"

    def compose(self) -> ComposeResult:
        with HorizontalScroll(id="namespace"):
            yield Label("Namespace:")
            yield Input(
                "sparse_test",
                placeholder="Type the namespace you want to query",
                max_length=128,
                id="namespace-input",
            )
        with HorizontalScroll(id="query"):
            yield Label("Query:")
            yield Input(
                placeholder="Type your query here...", max_length=512, id="query-input"
            )

    def on_mount(self):
        query = self.query_one("#query")
        query.get_child_by_type(Input).focus()

    def clear(self):
        query = self.query_one("#query")
        user_input = query.get_child_by_type(Input)
        user_input.value = ""
        user_input.focus()


class DocTable(ScrollableContainer):
    BORDER_TITLE = "Retrieve Results"

    def compose(self) -> ComposeResult:
        yield Static(id="elapsed")
        yield DataTable()

    def on_mount(self):
        table = self.query_one(DataTable)
        table.cursor_type = "row"

    def fill_retrieve(self, retrieve: RetrieveResponse):
        time = self.query_one(Static)
        time.update(f"Time cost: {retrieve.elapsed:.6f}")
        table = self.query_one(DataTable)
        if not table.columns:
            table.add_columns(*RETRIEVE_COLUMNS)
        for doc in retrieve.docs:
            table.add_row(doc.id, doc.title, doc.text)

    def fill_rank(self, ranked: RankedResponse):
        def pretty_index_value(value: int | None, base: int) -> str:
            if value is None:
                return "❌"
            if value > base:
                return f"🔼{value}"
            elif value < base:
                return f"🔽{value}"
            return f"🔵{value}"

        time = self.query_one(Static)
        time.update(f"Time cost: {ranked.elapsed:.6f}")
        table = self.query_one(DataTable)
        if not table.columns:
            table.add_columns(*RANKED_COLUMNS)
        for i, (doc, from_vec, from_sparse, from_text) in enumerate(
            zip(ranked.docs, ranked.from_vector, ranked.from_sparse, ranked.from_text)
        ):
            table.add_row(
                i,
                pretty_index_value(from_vec, i),
                pretty_index_value(from_sparse, i),
                pretty_index_value(from_text, i),
                doc.id,
                doc.title,
                doc.text,
            )

    def clear(self):
        time = self.query_one(Static)
        time.update()
        table = self.query_one(DataTable)
        table.clear(columns=True)


class QueryApp(App):
    BINDINGS: ClassVar[list] = [
        ("ctrl+r", "clear_input", "Clear the input field"),
        ("ctrl+d", "toggle_dark", "Toggle dark mode"),
    ]
    CSS_PATH = "tui.tcss"

    def __init__(self, port):
        self.client = QTextAsyncClient(port=port)
        super().__init__()

    def action_clear_input(self):
        form = self.query_one(Form)
        form.clear()

        tables = self.query(DocTable)
        for table in tables.nodes:
            table.clear()

    def action_toggle_dark(self):
        self.dark = not self.dark

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        yield Form()
        with TabbedContent(initial="ranked"):
            with TabPane("Ranked", id="ranked"):
                yield DocTable()
            with TabPane("Vector", id="vector"):
                yield DocTable()
            with TabPane("Sparse", id="sparse"):
                yield DocTable()
            with TabPane("Text", id="text"):
                yield DocTable()

    @on(Input.Submitted)
    async def query_server(self, event: Input.Submitted):
        namespace = self.query_one("#namespace-input")
        query = self.query_one("#query-input")
        if query.value and namespace.value:
            self.run_worker(
                self.hybrid_search(namespace.value, query.value), exclusive=True
            )
        else:
            self.notify("Please input the query and namespace")

    async def hybrid_search(self, namespace: str, query: str):
        try:
            explain = await self.client.query_explain(namespace, query)
        except HTTPStatusError as err:
            self.notify(
                err.response.text,
                severity="error",
                title=f"Request Error: {err.response.status_code}",
            )
            return
        tabs = self.query(TabPane)
        for tab in tabs.nodes:
            table = tab.get_child_by_type(DocTable)
            docs: RetrieveResponse = getattr(explain, tab.id)
            if tab.id == "ranked":
                table.fill_rank(docs)
            else:
                table.fill_retrieve(docs)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        PORT = int(sys.argv[1])
    app = QueryApp(port=PORT)
    app.run()
