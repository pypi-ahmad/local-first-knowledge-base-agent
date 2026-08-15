"""Read Chromium (Chrome/Edge) browser history for indexing.

The History file is a live SQLite DB the browser keeps locked, so we copy
it to a temp file before opening. Both browsers share the same schema.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from config import BROWSER_HISTORY_PATHS

# Chrome/WebKit timestamps are microseconds since 1601-01-01 UTC.
_WEBKIT_EPOCH_OFFSET_SECONDS = 11644473600


class HistoryEntry(TypedDict):
    url: str
    title: str
    visited_at: str  # ISO date
    visit_count: int
    browser: str


def _chrome_time_to_iso_date(chrome_time: int) -> str:
    epoch_seconds = (chrome_time / 1_000_000) - _WEBKIT_EPOCH_OFFSET_SECONDS
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).date().isoformat()


def available_browsers() -> list[str]:
    return [name for name, path in BROWSER_HISTORY_PATHS.items() if path.is_file()]


def read_history(history_path: Path, browser: str, limit: int = 5000) -> list[HistoryEntry]:
    if not history_path.is_file():
        return []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_copy = Path(tmp_dir) / "History_copy"
        shutil.copy2(history_path, tmp_copy)

        conn = sqlite3.connect(f"file:{tmp_copy}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                """
                SELECT url, title, last_visit_time, visit_count
                FROM urls
                WHERE last_visit_time > 0
                ORDER BY last_visit_time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()

    return [
        HistoryEntry(
            url=url,
            title=title or url,
            visited_at=_chrome_time_to_iso_date(last_visit_time),
            visit_count=visit_count,
            browser=browser,
        )
        for url, title, last_visit_time, visit_count in rows
    ]


def read_all_available(limit_per_browser: int = 5000) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    for browser, path in BROWSER_HISTORY_PATHS.items():
        if path.is_file():
            entries.extend(read_history(path, browser, limit_per_browser))
    return entries
