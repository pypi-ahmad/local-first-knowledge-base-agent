"""SQLite-backed metadata store for incremental indexing.

Tracks one row per indexed file (mtime + content hash) so re-scans only
touch files that actually changed.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

from config import METADATA_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    source_type TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS excluded_paths (
    path TEXT PRIMARY KEY,
    added_at TEXT NOT NULL
);
"""


@dataclass
class FileRecord:
    path: str
    mtime: float
    size: int
    content_hash: str
    source_type: str
    chunk_count: int
    indexed_at: str


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


def get_file_record(path: str) -> Optional[FileRecord]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        return FileRecord(**dict(row)) if row else None


def is_unchanged(path: str, mtime: float, size: int) -> bool:
    """Cheap pre-check (no hashing) before doing a full content-hash comparison."""
    record = get_file_record(path)
    return record is not None and record.mtime == mtime and record.size == size


def upsert_file_record(
    path: str, mtime: float, size: int, content_hash: str, source_type: str, chunk_count: int
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO files (path, mtime, size, content_hash, source_type, chunk_count, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                mtime=excluded.mtime, size=excluded.size, content_hash=excluded.content_hash,
                source_type=excluded.source_type, chunk_count=excluded.chunk_count,
                indexed_at=excluded.indexed_at
            """,
            (path, mtime, size, content_hash, source_type, chunk_count, datetime.now(timezone.utc).isoformat()),
        )


def delete_file_record(path: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM files WHERE path = ?", (path,))


def list_files(source_type: Optional[str] = None) -> list[FileRecord]:
    with _connect() as conn:
        if source_type:
            rows = conn.execute("SELECT * FROM files WHERE source_type = ?", (source_type,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM files").fetchall()
        return [FileRecord(**dict(r)) for r in rows]


def get_stats() -> dict:
    with _connect() as conn:
        total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        total_chunks = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) FROM files").fetchone()[0]
        by_type = {
            row["source_type"]: row["c"]
            for row in conn.execute("SELECT source_type, COUNT(*) AS c FROM files GROUP BY source_type").fetchall()
        }
        last_indexed = conn.execute("SELECT MAX(indexed_at) FROM files").fetchone()[0]
        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "by_type": by_type,
            "last_indexed_at": last_indexed,
        }


def clear_all() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM files")


def exclude_path(path: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO excluded_paths (path, added_at) VALUES (?, ?)",
            (path, datetime.now(timezone.utc).isoformat()),
        )


def unexclude_path(path: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM excluded_paths WHERE path = ?", (path,))


def list_excluded_paths() -> list[str]:
    with _connect() as conn:
        return [r["path"] for r in conn.execute("SELECT path FROM excluded_paths").fetchall()]


def is_excluded(path: str, excluded: Optional[list[str]] = None) -> bool:
    excluded = excluded if excluded is not None else list_excluded_paths()
    return any(path == p or path.startswith(p.rstrip("/\\") + "/") or path.startswith(p.rstrip("/\\") + "\\") for p in excluded)


def _demo() -> None:
    """Smoke check: the mtime/size fast-path must reject any real content change."""
    upsert_file_record("demo.md", mtime=100.0, size=10, content_hash="abc", source_type="note", chunk_count=2)
    assert is_unchanged("demo.md", 100.0, 10)
    assert not is_unchanged("demo.md", 100.0, 11)
    assert not is_unchanged("missing.md", 1.0, 1)
    stats = get_stats()
    assert stats["total_files"] >= 1
    delete_file_record("demo.md")
    print("db.metadata self-check ok")


if __name__ == "__main__":
    _demo()
