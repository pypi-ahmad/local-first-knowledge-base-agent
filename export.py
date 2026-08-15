"""Export answers/reports as Markdown or PDF. Pure formatting — no LLM calls.

ponytail: PDF via PyMuPDF's Story/DocumentWriter (already a project dependency
for PDF reading) instead of adding reportlab/weasyprint for the write side too.
"""

from __future__ import annotations

import html as html_module
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from state import Citation


def build_export_markdown(title: str, body: str, citations: list[Citation] | None = None) -> str:
    lines = [f"# {title}", "", body]
    if citations:
        lines += ["", "## Sources"]
        for i, c in enumerate(citations, start=1):
            lines.append(f"{i}. `{c['source']}` ({c['source_type']}, {c.get('date') or 'undated'}): {c['snippet']}")
    lines += ["", f"_Exported {datetime.now(timezone.utc).date().isoformat()}_"]
    return "\n".join(lines)


def _markdown_to_simple_html(markdown_text: str) -> str:
    paragraphs = markdown_text.split("\n\n")
    body = "".join(
        f"<p>{html_module.escape(p).replace(chr(10), '<br/>')}</p>" for p in paragraphs if p.strip()
    )
    return f"<html><body style='font-family:sans-serif'>{body}</body></html>"


def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    """DocumentWriter needs a real file path to infer the "pdf" format (a
    plain in-memory stream has no extension to sniff), so we write to a
    temp file and read the bytes back."""
    html = _markdown_to_simple_html(markdown_text)
    story = pymupdf.Story(html=html)
    mediabox = pymupdf.paper_rect("letter")
    where = mediabox + (36, 36, -36, -36)

    # ignore_cleanup_errors: mupdf's writer can hold its file handle open past
    # .close() on Windows; we already have the bytes we need by then.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir) / "export.pdf"
        writer = pymupdf.DocumentWriter(str(tmp_path))
        more = True
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(device)
            writer.end_page()
        writer.close()
        del writer
        return tmp_path.read_bytes()
