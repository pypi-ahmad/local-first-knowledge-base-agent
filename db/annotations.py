"""Decisions/action-items/open-questions extracted during indexing, plus
pinned Q&A answers from the chat UI.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from config import METADATA_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    entity TEXT,
    source TEXT NOT NULL,
    date TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pinned_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(METADATA_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_annotation(kind: str, text: str, source: str, entity: Optional[str] = None, date: Optional[str] = None) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO annotations (kind, text, entity, source, date, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (kind, text, entity, source, date, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def list_annotations(kind: Optional[str] = None, since: Optional[str] = None, entity: Optional[str] = None) -> list[dict]:
    query = "SELECT * FROM annotations WHERE 1=1"
    params: list = []
    if kind:
        query += " AND kind = ?"
        params.append(kind)
    if since:
        query += " AND date >= ?"
        params.append(since)
    if entity:
        query += " AND entity = ?"
        params.append(entity)
    query += " ORDER BY date DESC"
    with _connect() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def mark_status(annotation_id: int, status: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE annotations SET status = ? WHERE id = ?", (status, annotation_id))


def delete_annotations_by_source(source: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM annotations WHERE source = ?", (source,))


def pin_answer(question: str, answer: str, citations: list) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO pinned_answers (question, answer, citations_json, created_at) VALUES (?, ?, ?, ?)",
            (question, answer, json.dumps(citations), datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def list_pinned() -> list[dict]:
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM pinned_answers ORDER BY created_at DESC").fetchall()]
    for row in rows:
        row["citations"] = json.loads(row.pop("citations_json"))
    return rows


def unpin(pinned_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM pinned_answers WHERE id = ?", (pinned_id,))


def clear_all() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM annotations")
        conn.execute("DELETE FROM pinned_answers")


def _demo() -> None:
    clear_all()
    aid = add_annotation("decision", "Ship v1 with Chroma", "notes/plan.md", entity="Project Atlas", date="2026-07-10")
    assert len(list_annotations(kind="decision")) == 1
    mark_status(aid, "outdated")
    assert list_annotations()[0]["status"] == "outdated"
    pid = pin_answer("What did we decide?", "Ship v1 with Chroma.", [{"source": "notes/plan.md"}])
    assert list_pinned()[0]["citations"][0]["source"] == "notes/plan.md"
    unpin(pid)
    assert list_pinned() == []
    clear_all()
    print("db.annotations self-check ok")


if __name__ == "__main__":
    _demo()
