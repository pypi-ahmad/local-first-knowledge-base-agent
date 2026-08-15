"""Incremental indexing orchestration: scan -> load -> chunk -> extract ->
embed+store -> track metadata. Handles notes/code/PDFs/images/audio and
browser history, plus deletions of files removed from a watched folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

import utils
from agents import extraction
from db import annotations as annotations_db
from db import graph_store, metadata
from indexer import audio_loader, browser_history, code_parser, image_loader, loaders
from retriever import store

ProgressCallback = Callable[[dict], None]


def _extract_and_store_graph(llm: Optional[BaseChatModel], chunk: str, source: str, date_str: str) -> None:
    if llm is None:
        return
    try:
        result = extraction.extract_from_chunk(llm, chunk)
    except Exception:
        return
    for entity in result.entities:
        graph_store.upsert_entity(entity.name, entity.kind, date_str)
    for relation in result.relations:
        graph_store.upsert_relation(relation.source, relation.target, relation.relation, source, date_str)
    for ann in result.annotations:
        annotations_db.add_annotation(ann.kind, ann.text, source, entity=ann.entity, date=date_str)


def _load_chunks(path: Path, source_type: str) -> list[dict]:
    """Returns [{"content": str, "symbol": Optional[str], "kind": Optional[str]}, ...]."""
    if source_type == "image":
        text = image_loader.ocr_caption_image(path)
        return [{"content": text, "symbol": None, "kind": None}] if text.strip() else []

    if source_type == "audio":
        text = audio_loader.transcribe_audio(path)
        return [{"content": c, "symbol": None, "kind": None} for c in loaders.chunk_text(text)]

    text = loaders.load_text(path)
    if source_type == "code" and path.suffix.lower() == ".py":
        parsed = code_parser.parse_python_chunks(text)
        if parsed:
            chunks = []
            for item in parsed:
                for sub in loaders.chunk_text(item["content"]):
                    chunks.append({"content": sub, "symbol": item["symbol"], "kind": item["kind"]})
            return chunks

    return [{"content": c, "symbol": None, "kind": None} for c in loaders.chunk_text(text)]


def _remove_file(path_str: str, vs) -> None:
    store.delete_by_source(vs, path_str)
    graph_store.delete_relations_by_source_doc(path_str)
    annotations_db.delete_annotations_by_source(path_str)
    metadata.delete_file_record(path_str)


def _index_single_file(path: Path, vs, llm: Optional[BaseChatModel], deep_extraction: bool) -> int:
    """Returns the number of chunks written."""
    path_str = str(path)
    source_type = loaders.source_type_for(path)
    stat = path.stat()
    date_str = utils.epoch_to_iso_date(stat.st_mtime)
    folder_str = str(path.parent)

    chunks = _load_chunks(path, source_type)

    # Clear any prior chunks/graph/annotations for this file before re-adding.
    store.delete_by_source(vs, path_str)
    graph_store.delete_relations_by_source_doc(path_str)
    annotations_db.delete_annotations_by_source(path_str)

    texts, metadatas, ids = [], [], []
    for idx, item in enumerate(chunks):
        chunk_id = f"{path_str}::{idx}"
        meta = {
            "source": path_str,
            "source_type": source_type,
            "date": date_str,
            "date_ordinal": utils.iso_date_to_ordinal(date_str),
            "folder": folder_str,
            "chunk_id": chunk_id,
            "chunk_index": idx,
        }
        if item["symbol"]:
            meta["symbol"] = item["symbol"]
        if item["kind"]:
            meta["code_kind"] = item["kind"]
        texts.append(item["content"])
        metadatas.append(meta)
        ids.append(chunk_id)

        if deep_extraction:
            _extract_and_store_graph(llm, item["content"], path_str, date_str)

    store.add_documents(vs, texts, metadatas, ids)
    content_hash = utils.sha256_text("".join(texts)) if texts else utils.sha256_file(path)
    metadata.upsert_file_record(path_str, stat.st_mtime, stat.st_size, content_hash, source_type, len(texts))
    return len(texts)


def index_folder(
    folder: Path,
    embeddings: Embeddings,
    llm: Optional[BaseChatModel] = None,
    deep_extraction: bool = True,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict:
    vs = store.get_vectorstore(embeddings)
    excluded = metadata.list_excluded_paths()
    all_files = list(loaders.scan_folder(folder))
    total = len(all_files)
    stats = {"added": 0, "updated": 0, "skipped": 0, "removed": 0, "errors": []}
    found_paths: set[str] = set()

    for i, path in enumerate(all_files):
        path_str = str(path)
        found_paths.add(path_str)
        if metadata.is_excluded(path_str, excluded):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue

        if metadata.is_unchanged(path_str, stat.st_mtime, stat.st_size):
            stats["skipped"] += 1
        else:
            existing = metadata.get_file_record(path_str)
            try:
                _index_single_file(path, vs, llm, deep_extraction)
                stats["updated" if existing else "added"] += 1
            except Exception as exc:  # noqa: BLE001 - indexing must not abort on one bad file
                stats["errors"].append(f"{path_str}: {exc}")

        if progress_cb:
            progress_cb({"current": i + 1, "total": total, "file": path_str})

    folder_str = str(folder)
    for record in metadata.list_files():
        if record.path.startswith(folder_str) and record.path not in found_paths and not record.source_type == "history":
            _remove_file(record.path, vs)
            stats["removed"] += 1

    return stats


def index_browser_history(embeddings: Embeddings, progress_cb: Optional[ProgressCallback] = None) -> dict:
    vs = store.get_vectorstore(embeddings)
    entries = browser_history.read_all_available()
    stats = {"added": 0, "skipped": 0}

    for i, entry in enumerate(entries):
        synthetic_path = f"history::{entry['browser']}::{entry['url']}"
        content_hash = utils.sha256_text(f"{entry['title']}|{entry['visited_at']}|{entry['visit_count']}")
        existing = metadata.get_file_record(synthetic_path)
        if existing and existing.content_hash == content_hash:
            stats["skipped"] += 1
            continue

        text = f"{entry['title']}\n{entry['url']}"
        chunk_id = f"{synthetic_path}::0"
        store.delete_by_source(vs, synthetic_path)
        store.add_documents(
            vs,
            [text],
            [{
                "source": synthetic_path,
                "source_type": "history",
                "date": entry["visited_at"],
                "date_ordinal": utils.iso_date_to_ordinal(entry["visited_at"]),
                "folder": entry["browser"],
                "chunk_id": chunk_id,
                "chunk_index": 0,
                "url": entry["url"],
            }],
            [chunk_id],
        )
        metadata.upsert_file_record(synthetic_path, 0.0, len(text), content_hash, "history", 1)
        stats["added"] += 1

        if progress_cb:
            progress_cb({"current": i + 1, "total": len(entries), "file": entry["url"]})

    return stats


def clear_index(embeddings: Embeddings) -> None:
    vs = store.get_vectorstore(embeddings)
    store.clear_all(vs)
    metadata.clear_all()
    graph_store.clear_all()
    annotations_db.clear_all()


def purge_source(embeddings: Embeddings, source: str) -> None:
    vs = store.get_vectorstore(embeddings)
    _remove_file(source, vs)
