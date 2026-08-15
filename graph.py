"""LangGraph workflow wiring: query understanding -> retrieval -> rerank ->
generation -> citation -> reflection. Node logic lives in agents/nodes.py;
this module only builds and compiles the graph.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    citation_node,
    generation_node,
    query_understanding_node,
    rerank_node,
    retrieval_node,
    reflection_node,
)
from db.checkpointer import get_checkpointer
from state import KBState

_compiled_graph = None


def build_graph():
    graph = StateGraph(KBState)
    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generation", generation_node)
    graph.add_node("citation", citation_node)
    graph.add_node("reflection", reflection_node)

    graph.add_edge(START, "query_understanding")
    graph.add_edge("query_understanding", "retrieval")
    graph.add_edge("retrieval", "rerank")
    graph.add_edge("rerank", "generation")
    graph.add_edge("generation", "citation")
    graph.add_edge("citation", "reflection")
    graph.add_edge("reflection", END)

    return graph.compile(checkpointer=get_checkpointer())


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
