"""Graph-augmented retrieval: find entities mentioned in the query, traverse
the knowledge graph a few hops, and surface which additional source documents
are connected — even if they don't lexically/semantically match the query
text directly. The retrieval node then pulls real chunks from those sources
so answers can connect information across documents.
"""

from __future__ import annotations

import re

import config
from db import graph_store


def find_query_entities(query: str, limit: int = 5) -> list[str]:
    """Heuristic: capitalized words/phrases in the query are candidate entity
    mentions; match them against known graph entity names."""
    candidate_terms = set(re.findall(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", query))
    matches: list[str] = []
    for term in candidate_terms:
        matches.extend(graph_store.search_entities(term, limit=3))
    # de-dupe, preserve order
    seen: set[str] = set()
    unique = [m for m in matches if not (m in seen or seen.add(m))]
    return unique[:limit]


def graph_augmented_sources(query: str, hops: int = config.GRAPH_HOP_DEPTH) -> list[str]:
    """Return source-document paths connected to the query's entities within `hops`."""
    entities = find_query_entities(query)
    if not entities:
        return []
    edges = graph_store.neighbors_with_context(entities, hops=hops)
    return sorted({e["source_doc"] for e in edges if e.get("source_doc")})
