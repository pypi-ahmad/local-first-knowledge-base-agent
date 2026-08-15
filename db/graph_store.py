"""Lightweight local knowledge graph: entities + relations in sqlite, assembled
into an in-memory networkx graph for traversal (GraphRAG-style retrieval).

sqlite is the single source of truth; networkx is only ever a rebuilt view
over it, so there is no second persistence format to keep in sync.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

import networkx as nx

from config import METADATA_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    relation TEXT NOT NULL,
    source_doc TEXT NOT NULL,
    date TEXT,
    UNIQUE(source_entity, target_entity, relation, source_doc)
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


def upsert_entity(name: str, kind: str, date: Optional[str] = None) -> None:
    date = date or datetime.now(timezone.utc).date().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO entities (name, kind, first_seen, last_seen, mention_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(name) DO UPDATE SET
                last_seen = MAX(last_seen, excluded.last_seen),
                first_seen = MIN(first_seen, excluded.first_seen),
                mention_count = mention_count + 1
            """,
            (name, kind, date, date),
        )


def upsert_relation(source_entity: str, target_entity: str, relation: str, source_doc: str, date: Optional[str] = None) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO relations (source_entity, target_entity, relation, source_doc, date)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_entity, target_entity, relation, source_doc, date),
        )


def search_entities(term: str, limit: int = 10) -> list[str]:
    """Case-insensitive substring match against known entity names."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name FROM entities WHERE name LIKE ? ORDER BY mention_count DESC LIMIT ?",
            (f"%{term}%", limit),
        ).fetchall()
        return [r["name"] for r in rows]


def delete_relations_by_source_doc(source_doc: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM relations WHERE source_doc = ?", (source_doc,))


def most_mentioned_since(date_start: str, limit: int = 5) -> list[tuple[str, int]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name, mention_count FROM entities WHERE last_seen >= ? ORDER BY mention_count DESC LIMIT ?",
            (date_start, limit),
        ).fetchall()
        return [(r["name"], r["mention_count"]) for r in rows]


def entity_first_seen(name: str) -> Optional[str]:
    with _connect() as conn:
        row = conn.execute("SELECT first_seen FROM entities WHERE name = ?", (name,)).fetchone()
        return row["first_seen"] if row else None


def build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    with _connect() as conn:
        for row in conn.execute("SELECT name, kind FROM entities"):
            graph.add_node(row["name"], kind=row["kind"])
        for row in conn.execute("SELECT source_entity, target_entity, relation, source_doc, date FROM relations"):
            graph.add_edge(
                row["source_entity"],
                row["target_entity"],
                relation=row["relation"],
                source_doc=row["source_doc"],
                date=row["date"],
            )
    return graph


def neighbors_with_context(entity_names: list[str], hops: int = 2) -> list[dict]:
    """BFS out from the given entities up to `hops` steps. Returns one dict per
    traversed edge: {source, target, relation, source_doc, date}."""
    graph = build_graph()
    undirected = graph.to_undirected(as_view=True)
    seen_nodes: set[str] = set()
    frontier = [n for n in entity_names if n in undirected]
    for _ in range(hops):
        next_frontier = []
        for node in frontier:
            for neighbor in undirected.neighbors(node):
                if neighbor not in seen_nodes:
                    next_frontier.append(neighbor)
        seen_nodes.update(frontier)
        frontier = next_frontier
    seen_nodes.update(frontier)

    edges = []
    for source, target, data in graph.edges(data=True):
        if source in seen_nodes or target in seen_nodes:
            edges.append({"source": source, "target": target, **data})
    return edges


def clear_all() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM entities")
        conn.execute("DELETE FROM relations")


def _demo() -> None:
    clear_all()
    upsert_entity("Project Atlas", "project", "2026-01-05")
    upsert_entity("Ahmad", "person", "2026-01-05")
    upsert_relation("Ahmad", "Project Atlas", "owns", "notes/atlas.md", "2026-01-05")
    assert "Project Atlas" in search_entities("atlas")
    edges = neighbors_with_context(["Ahmad"], hops=1)
    assert any(e["target"] == "Project Atlas" for e in edges)
    assert entity_first_seen("Ahmad") == "2026-01-05"
    clear_all()
    print("db.graph_store self-check ok")


if __name__ == "__main__":
    _demo()
