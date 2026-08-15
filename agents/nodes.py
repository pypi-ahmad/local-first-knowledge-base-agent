"""LangGraph node functions: query understanding -> retrieval -> rerank ->
generation -> citation -> reflection.

Query expansion and re-ranking always use a local Ollama model regardless of
which provider/model the user picked for the final answer — they're
retrieval-quality helpers, not the user-facing generation step, so they stay
local per the app's "local models for the majority of embedding, retrieval,
and generation" mandate.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

import config
from agents import models
from retriever import graph_rag, keyword, rerank as rerank_module, store, temporal
from state import Citation, KBState, RetrievedDoc
from utils import extract_text, make_snippet

_SOURCE_TYPE_KEYWORDS = {
    "codebase": "code", "code": "code",
    "pdf": "pdf", "pdfs": "pdf",
    "notes": "note", "note": "note",
    "images": "image", "image": "image", "photo": "image", "screenshot": "image",
    "audio": "audio", "recording": "audio", "voice memo": "audio",
    "history": "history", "browser": "history", "visited": "history",
}

_SYSTEM_PROMPT = (
    "You are a personal knowledge-base assistant. Answer using ONLY the "
    "provided context passages. Always cite passages by their [n] marker "
    "inline. If the context doesn't contain the answer, say so plainly "
    "instead of guessing."
)

_ANSWER_PROMPT = "Context passages:\n{context}\n\nQuestion: {query}\n\nAnswer using the context above, citing [n] markers inline."


class _Expansion(BaseModel):
    paraphrases: list[str]


class _ReflectionResult(BaseModel):
    sufficient: bool
    notes: str = ""


def _local_llm(state: KBState) -> Optional[BaseChatModel]:
    if state["provider"] == "ollama":
        return models.build_chat_model("ollama", state["model"])
    available = models.list_ollama_models()
    if not available:
        return None
    return models.build_chat_model("ollama", available[0])


def _detect_source_type_hint(query: str) -> Optional[str]:
    q = query.lower()
    for keyword_, source_type in _SOURCE_TYPE_KEYWORDS.items():
        if keyword_ in q:
            return source_type
    return None


def query_understanding_node(state: KBState) -> dict:
    query = state["messages"][-1].content
    filters = dict(state.get("filters") or {})

    date_range = temporal.parse_date_range(query)
    if date_range:
        start, end = date_range
        if start:
            filters["date_start"] = start
        if end:
            filters["date_end"] = end

    if "file_types" not in filters:
        hint = _detect_source_type_hint(query)
        if hint:
            filters["file_types"] = [hint]

    return {"query": query, "filters": filters}


def _expand_queries(state: KBState, query: str) -> list[str]:
    llm = _local_llm(state)
    if llm is None:
        return [query]
    try:
        structured_llm = llm.with_structured_output(_Expansion)
        result = structured_llm.invoke(
            f"Generate {config.QUERY_EXPANSION_COUNT} different paraphrases of this "
            f"question that preserve its meaning, to improve search recall:\n\n{query}"
        )
        return [query, *result.paraphrases[: config.QUERY_EXPANSION_COUNT]]
    except Exception:
        return [query]


def retrieval_node(state: KBState) -> dict:
    embeddings = models.build_embeddings(state["embedding_model"])
    vs = store.get_vectorstore(embeddings)
    where = store.build_where_filter(state["filters"])

    candidates: dict[str, RetrievedDoc] = {}
    for expanded_query in _expand_queries(state, state["query"]):
        for doc in keyword.hybrid_search(vs, expanded_query, k=config.RETRIEVAL_TOP_K, where=where):
            key = doc["id"] or f"{doc['source']}::{doc['content'][:50]}"
            if key not in candidates or doc["score"] > candidates[key]["score"]:
                candidates[key] = doc

    graph_sources = graph_rag.graph_augmented_sources(state["query"])
    if graph_sources:
        extra_where = {"source": {"$in": graph_sources}}
        if where:
            extra_where = {"$and": [where, extra_where]}
        for doc in store.vector_search(vs, state["query"], k=10, where=extra_where):
            key = doc["id"] or f"{doc['source']}::{doc['content'][:50]}"
            candidates.setdefault(key, doc)

    ranked = sorted(candidates.values(), key=lambda d: d["score"], reverse=True)[: config.RETRIEVAL_TOP_K]
    return {"retrieved_docs": ranked}


def rerank_node(state: KBState) -> dict:
    llm = _local_llm(state)
    if llm is None:
        return {"reranked_docs": state["retrieved_docs"][: config.RERANK_TOP_K]}
    reranked = rerank_module.rerank(llm, state["query"], state["retrieved_docs"], top_k=config.RERANK_TOP_K)
    return {"reranked_docs": reranked}


def format_context(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "(no relevant documents found in the knowledge base)"
    return "\n\n".join(
        f"[{i}] Source: {d['source']} (type={d['source_type']}, date={d['date']})\n{d['content'][:800]}"
        for i, d in enumerate(docs, start=1)
    )


def generation_node(state: KBState) -> dict:
    llm = models.build_chat_model(state["provider"], state["model"], local_only=state.get("local_only", False))
    context = format_context(state["reranked_docs"])
    history = list(state["messages"][:-1])
    prompt_messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        *history,
        HumanMessage(content=_ANSWER_PROMPT.format(context=context, query=state["query"])),
    ]
    response = llm.invoke(prompt_messages)
    answer = extract_text(response.content)
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


def citation_node(state: KBState) -> dict:
    citations: list[Citation] = []
    seen_sources: set[str] = set()
    for doc in state["reranked_docs"]:
        if doc["source"] in seen_sources:
            continue
        seen_sources.add(doc["source"])
        citations.append(
            Citation(
                source=doc["source"],
                snippet=make_snippet(doc["content"], state["query"]),
                date=doc["date"],
                source_type=doc["source_type"],
            )
        )
    return {"citations": citations}


def reflection_node(state: KBState) -> dict:
    """Informational quality check only — does not loop/retry (ponytail: a
    real retry loop risks unbounded cost; this just flags low-confidence answers)."""
    llm = _local_llm(state)
    if llm is None:
        return {"needs_retry": False, "reflection_notes": None}
    try:
        structured_llm = llm.with_structured_output(_ReflectionResult)
        result = structured_llm.invoke(
            f"Question: {state['query']}\n\nAnswer given: {state['answer']}\n\n"
            "Does this answer fully and accurately address the question? Flag it if "
            "it seems incomplete, hedged, or says the information wasn't found."
        )
        return {"needs_retry": not result.sufficient, "reflection_notes": result.notes or None}
    except Exception:
        return {"needs_retry": False, "reflection_notes": None}
