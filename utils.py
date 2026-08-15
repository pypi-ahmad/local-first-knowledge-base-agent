"""Small shared helpers: hashing, snippets, dates. No project-specific logic."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def make_snippet(text: str, query: str = "", length: int = 240) -> str:
    """Return a short excerpt centered on the first query term match, else the start."""
    text = " ".join(text.split())
    if not text:
        return ""
    idx = -1
    for term in query.split():
        idx = text.lower().find(term.lower())
        if idx != -1:
            break
    if idx == -1:
        excerpt = text[:length]
    else:
        start = max(0, idx - length // 2)
        excerpt = text[start : start + length]
    prefix = "..." if idx > length // 2 else ""
    suffix = "..." if len(text) > length else ""
    return f"{prefix}{excerpt}{suffix}"


def epoch_to_iso_date(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).date().isoformat()


def iso_date_to_ordinal(iso_date: str) -> int:
    """YYYY-MM-DD -> sortable int (20260715). Chroma's $gte/$lte only accept
    numbers, not date strings, so range filters compare on this instead."""
    return int(iso_date.replace("-", ""))


def extract_text(content) -> str:
    """LangChain message .content is usually a str, but some providers (e.g.
    Gemini) return a list of content-part dicts instead. Pull just the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)
