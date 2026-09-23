"""Compact HTML table serialization and strict table parser."""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from html.parser import HTMLParser


def _html_table(
    table_id: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> list[str]:
    lines = [f'<table id="{table_id}">', _html_row("th", headers)]
    lines.extend(_html_row("td", row) for row in rows)
    lines.append("</table>")
    return lines


def _html_row(tag: str, values: Sequence[object]) -> str:
    return "<tr>" + "".join(f"<{tag}>{html.escape(str(value), quote=True)}" for value in values)


class _CompactHtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.schema: str | None = None
        self.tables: dict[str, list[list[str]]] = {}
        self._table: str | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "main":
            if self.schema is not None:
                raise ValueError("duplicate HTML main element")
            self.schema = attributes.get("data-schema")
        elif tag == "table":
            self._finish_table()
            table_id = attributes.get("id")
            if not table_id or table_id in self.tables:
                raise ValueError(f"invalid or duplicate HTML table: {table_id!r}")
            self._table = table_id
            self.tables[table_id] = []
        elif tag == "tr":
            self._finish_row()
            if self._table is None:
                raise ValueError("HTML row outside table")
            self._row = []
        elif tag in {"th", "td"}:
            self._finish_cell()
            if self._row is None:
                raise ValueError("HTML cell outside row")
            self._cell = []
        elif tag not in {"meta"}:
            raise ValueError(f"unsupported HTML tag: {tag}")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"}:
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()
        elif tag == "table":
            self._finish_table()
        elif tag == "main":
            self._finish_table()
        else:
            raise ValueError(f"unsupported HTML closing tag: {tag}")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        elif data.strip():
            raise ValueError(f"unexpected HTML text: {data!r}")

    def close(self) -> None:
        super().close()
        self._finish_table()

    def _finish_cell(self) -> None:
        if self._cell is not None:
            if self._row is None:
                raise ValueError("HTML cell lost its row")
            self._row.append("".join(self._cell).strip())
            self._cell = None

    def _finish_row(self) -> None:
        self._finish_cell()
        if self._row is not None:
            if self._table is None:
                raise ValueError("HTML row lost its table")
            self.tables[self._table].append(self._row)
            self._row = None

    def _finish_table(self) -> None:
        self._finish_row()
        self._table = None


def _require_headers(rows: Sequence[Sequence[str]], expected: Sequence[str]) -> None:
    if not rows or tuple(rows[0]) != tuple(expected):
        raise ValueError(f"unexpected HTML headers: {rows[0] if rows else None}")
    for row in rows[1:]:
        if len(row) != len(expected):
            raise ValueError("HTML table row has wrong field count")
