"""BM25 keyword search and reciprocal-rank-fusion hybrid combination with
vector search. BM25 index is rebuilt per query from Chroma's stored docs.

ponytail: rebuild-per-query is O(n) over the collection; fine at personal-KB
scale (thousands of chunks). Persist/cache the index if the corpus grows large.
"""

from __future__ import annotations

from typing import Optional

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

import config
from state import RetrievedDoc

_RRF_K = 60


def bm25_search(vs: Chroma, query: str, k: int, where: Optional[dict] = None) -> list[RetrievedDoc]:
    data = vs.get(where=where, include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not documents:
        return []

    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0

    results: list[RetrievedDoc] = []
    for i in top_idx:
        meta = metadatas[i] or {}
        results.append(
            RetrievedDoc(
                id=meta.get("chunk_id", ""),
                content=documents[i],
                source=meta.get("source", ""),
                source_type=meta.get("source_type", ""),
                date=meta.get("date"),
                score=float(scores[i]) / max_score,
            )
        )
    return results


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedDoc]], weights: Optional[list[float]] = None
) -> list[RetrievedDoc]:
    """Merge multiple ranked result lists by reciprocal rank, keyed by chunk id."""
    weights = weights or [1.0] * len(ranked_lists)
    fused_scores: dict[str, float] = {}
    docs_by_id: dict[str, RetrievedDoc] = {}

    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs):
            key = doc["id"] or f"{doc['source']}::{doc['content'][:50]}"
            fused_scores[key] = fused_scores.get(key, 0.0) + weight / (rank + _RRF_K)
            docs_by_id.setdefault(key, doc)

    ranked_keys = sorted(fused_scores, key=lambda k: fused_scores[k], reverse=True)
    fused: list[RetrievedDoc] = []
    for key in ranked_keys:
        doc = dict(docs_by_id[key])
        doc["score"] = fused_scores[key]
        fused.append(RetrievedDoc(**doc))
    return fused


def hybrid_search(vs: Chroma, query: str, k: int, where: Optional[dict] = None) -> list[RetrievedDoc]:
    from retriever.store import vector_search

    vector_results = vector_search(vs, query, k=k * 2, where=where)
    keyword_results = bm25_search(vs, query, k=k * 2, where=where)
    fused = reciprocal_rank_fusion(
        [vector_results, keyword_results],
        weights=[config.HYBRID_VECTOR_WEIGHT, 1 - config.HYBRID_VECTOR_WEIGHT],
    )
    return fused[:k]
