"""Shared LangGraph state schema for the knowledge-base agent graph."""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class RetrievedDoc(TypedDict):
    id: str
    content: str
    source: str
    source_type: str
    date: Optional[str]
    score: float


class Citation(TypedDict):
    source: str
    snippet: str
    date: Optional[str]
    source_type: str


class SearchFilters(TypedDict, total=False):
    file_types: list[str]
    folders: list[str]
    date_start: Optional[str]
    date_end: Optional[str]


class KBState(TypedDict):
    messages: Annotated[list, add_messages]

    query: str
    provider: str
    model: str
    embedding_model: str
    local_only: bool

    filters: SearchFilters
    retrieved_docs: list[RetrievedDoc]
    reranked_docs: list[RetrievedDoc]

    answer: str
    citations: list[Citation]

    reflection_notes: Optional[str]
    needs_retry: bool
