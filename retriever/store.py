"""Local Chroma vector store: add/delete/search over indexed chunks."""

from __future__ import annotations

from typing import Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

import config
from state import RetrievedDoc, SearchFilters
from utils import iso_date_to_ordinal


def get_vectorstore(embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


def add_documents(vs: Chroma, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    if texts:
        vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def delete_by_source(vs: Chroma, source: str) -> None:
    vs.delete(where={"source": source})


def clear_all(vs: Chroma) -> None:
    existing = vs.get(include=[])
    ids = existing.get("ids") or []
    if ids:
        vs.delete(ids=ids)


def build_where_filter(filters: SearchFilters) -> Optional[dict]:
    clauses = []
    if filters.get("file_types"):
        clauses.append({"source_type": {"$in": filters["file_types"]}})
    if filters.get("folders"):
        clauses.append({"folder": {"$in": filters["folders"]}})
    if filters.get("date_start"):
        clauses.append({"date_ordinal": {"$gte": iso_date_to_ordinal(filters["date_start"])}})
    if filters.get("date_end"):
        clauses.append({"date_ordinal": {"$lte": iso_date_to_ordinal(filters["date_end"])}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def vector_search(vs: Chroma, query: str, k: int, where: Optional[dict] = None) -> list[RetrievedDoc]:
    # similarity_search_with_score returns raw distance (metric-dependent, unbounded);
    # 1/(1+distance) gives a stable (0, 1] "higher is better" score regardless of the
    # collection's distance function, avoiding langchain's relevance-score warning.
    results = vs.similarity_search_with_score(query, k=k, filter=where)
    docs: list[RetrievedDoc] = []
    for doc, distance in results:
        meta = doc.metadata or {}
        docs.append(
            RetrievedDoc(
                id=meta.get("chunk_id", ""),
                content=doc.page_content,
                source=meta.get("source", ""),
                source_type=meta.get("source_type", ""),
                date=meta.get("date"),
                score=1.0 / (1.0 + max(float(distance), 0.0)),
            )
        )
    return docs
