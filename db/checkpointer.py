"""Cross-session LangGraph conversation memory via a persistent sqlite checkpointer.

A module-level singleton: Streamlit reimports this module once per process
(reruns don't re-trigger imports), so the same connection is reused across
reruns and sessions, giving conversations memory across app restarts.
"""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from config import DATA_DIR

CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.sqlite3"

_saver: SqliteSaver | None = None


def get_checkpointer() -> SqliteSaver:
    global _saver
    if _saver is None:
        conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
        _saver = SqliteSaver(conn)
    return _saver
