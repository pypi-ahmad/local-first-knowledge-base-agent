"""File discovery, text extraction, and chunking for notes/code/PDFs/images/audio."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Iterator

import pymupdf as fitz

from config import (
    ALL_INDEXABLE_EXT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SUPPORTED_AUDIO_EXT,
    SUPPORTED_CODE_EXT,
    SUPPORTED_IMAGE_EXT,
    SUPPORTED_PDF_EXT,
)

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".codegraph", ".code-review-graph", "dist", "build"}


def scan_folder(folder: Path) -> Iterator[Path]:
    """Yield indexable file paths under folder, skipping VCS/dependency/build dirs."""
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in ALL_INDEXABLE_EXT:
            yield path


def source_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SUPPORTED_PDF_EXT:
        return "pdf"
    if ext in SUPPORTED_IMAGE_EXT:
        return "image"
    if ext in SUPPORTED_AUDIO_EXT:
        return "audio"
    if ext in SUPPORTED_CODE_EXT:
        return "code"
    return "note"


def load_text(path: Path) -> str:
    """Extract text for note/code/pdf files. Images and audio are handled by
    their own loader modules (network/model calls, not plain file reads)."""
    ext = path.suffix.lower()
    if ext in SUPPORTED_PDF_EXT:
        return _load_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header, *body = rows
    header_line = "| " + " | ".join(c or "" for c in header) + " |"
    sep_line = "| " + " | ".join("---" for _ in header) + " |"
    body_lines = ["| " + " | ".join(c or "" for c in row) + " |" for row in body]
    return "\n".join([header_line, sep_line, *body_lines])


def _load_pdf(path: Path) -> str:
    """Layout-aware extraction: headings (by relative font size) and tables
    (as markdown). Falls back to plain get_text() per page on any parse error,
    and to vision-model OCR for pages with no extractable text (scanned PDFs)."""
    doc = fitz.open(str(path))
    pages_text: list[str] = []
    try:
        for page in doc:
            pages_text.append(_extract_pdf_page(page))
    finally:
        doc.close()
    return "\n\n".join(p for p in pages_text if p)


def _extract_pdf_page(page: "fitz.Page") -> str:
    try:
        text = _extract_pdf_page_structured(page)
    except Exception:
        text = page.get_text() or ""

    if text.strip():
        return text

    # No extractable text layer -> likely a scanned page. Best-effort OCR via
    # the local vision model; silently skip if unavailable.
    try:
        from indexer.image_loader import ocr_caption_image_bytes

        pixmap = page.get_pixmap(dpi=200)
        return ocr_caption_image_bytes(pixmap.tobytes("png"))
    except Exception:
        return ""


def _extract_pdf_page_structured(page: "fitz.Page") -> str:
    page_dict = page.get_text("dict")
    sizes = [
        span["size"]
        for block in page_dict["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
    ]
    median_size = statistics.median(sizes) if sizes else 0.0

    lines: list[str] = []
    for block in page_dict["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            max_size = max((span["size"] for span in line["spans"]), default=0.0)
            lines.append(f"## {text}" if median_size and max_size >= median_size * 1.3 else text)

    try:
        for table in page.find_tables().tables:
            lines.append(_table_to_markdown(table.extract()))
    except Exception:
        pass

    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Greedy paragraph-aware splitter with character overlap between chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            # Single paragraph longer than chunk_size: hard-split with overlap.
            for start in range(0, len(para), chunk_size - overlap):
                chunks.append(para[start : start + chunk_size])
            current = ""
    if current:
        chunks.append(current)

    # Stitch overlap between adjacent chunks so retrieval doesn't lose boundary context.
    overlapped = [chunks[0]] if chunks else []
    for prev, curr in zip(chunks, chunks[1:]):
        tail = prev[-overlap:] if overlap else ""
        overlapped.append(f"{tail}{curr}" if tail else curr)
    return overlapped


def _demo() -> None:
    text = "para one " * 20 + "\n\n" + "para two " * 200 + "\n\n" + "para three"
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert all(len(c) <= 240 for c in chunks)
    assert chunk_text("") == []
    assert chunk_text("short") == ["short"]
    assert _table_to_markdown([]) == ""
    assert _table_to_markdown([["a", "b"], ["1", "2"]]) == "| a | b |\n| --- | --- |\n| 1 | 2 |"
    print("indexer.loaders self-check ok")


if __name__ == "__main__":
    _demo()
