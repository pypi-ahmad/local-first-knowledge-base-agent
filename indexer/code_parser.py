"""Structure-aware chunking for Python source via the stdlib `ast` module.

ponytail: Python only. Other languages fall back to the generic paragraph
chunker in loaders.py — add tree-sitter if structure-awareness for other
languages is actually needed.
"""

from __future__ import annotations

import ast


def parse_python_chunks(source: str) -> list[dict]:
    """Split Python source into one chunk per top-level function/class (full
    body incl. docstrings/comments/nested methods), plus one chunk for any
    leftover module-level code (imports, constants, top-level statements).
    Returns [] if the source doesn't parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    chunks: list[dict] = []
    covered_lines: set[int] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            segment = ast.get_source_segment(source, node)
            if not segment:
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append({"content": segment, "symbol": node.name, "kind": kind})
            covered_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    lines = source.splitlines()
    leftover = "\n".join(line for i, line in enumerate(lines, start=1) if i not in covered_lines).strip()
    if leftover:
        chunks.append({"content": leftover, "symbol": "<module>", "kind": "module"})
    return chunks


def _demo() -> None:
    src = (
        "import os\n\n"
        "def foo():\n    return 1\n\n"
        "class Bar:\n    def method(self):\n        pass\n"
    )
    chunks = parse_python_chunks(src)
    assert {c["kind"] for c in chunks} == {"function", "class", "module"}
    assert {c["symbol"] for c in chunks} == {"foo", "Bar", "<module>"}
    assert parse_python_chunks("def broken(:") == []
    print("indexer.code_parser self-check ok")


if __name__ == "__main__":
    _demo()
