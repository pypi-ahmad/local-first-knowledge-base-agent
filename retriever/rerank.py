"""Re-ranking of retrieved candidates before generation.

ponytail: LLM-based rerank (one batched structured call over all candidates)
instead of a cross-encoder model — avoids a multi-GB torch/sentence-transformers
install for a local-first app. Swap in a real cross-encoder if measured
precision is insufficient.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from state import RetrievedDoc

_SNIPPET_CHARS = 300


class _CandidateScore(BaseModel):
    index: int
    score: float = Field(ge=0, le=10)


class _RerankResult(BaseModel):
    scores: list[_CandidateScore]


_PROMPT = """Query: {query}

Rate how relevant each numbered passage is to answering the query, 0-10
(10 = directly answers it, 0 = irrelevant). Score every passage listed.

{listing}"""


def rerank(llm: BaseChatModel, query: str, candidates: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]:
    if not candidates:
        return []

    listing = "\n".join(f"[{i}] {c['content'][:_SNIPPET_CHARS]}" for i, c in enumerate(candidates))
    try:
        structured_llm = llm.with_structured_output(_RerankResult)
        result = structured_llm.invoke(_PROMPT.format(query=query, listing=listing))
        score_map = {s.index: s.score for s in result.scores}
    except Exception:
        return candidates[:top_k]

    scored = sorted(
        ((score_map.get(i, 0.0), c) for i, c in enumerate(candidates)),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [RetrievedDoc(**{**c, "score": score}) for score, c in scored[:top_k]]
