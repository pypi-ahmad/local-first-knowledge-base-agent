"""Proactive intelligence: on-demand digests, repeated-mention suggestions,
forgotten open questions, and conflicting-decision detection.

ponytail: digests are generated on demand (a UI button), not on a real OS
scheduler — Streamlit has no background cron. Conflict detection only compares
the two most recent decisions per entity, bounding LLM calls regardless of
how many decisions accumulate for that entity.
"""

from __future__ import annotations

from datetime import date, timedelta

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from db import annotations as annotations_db
from db import graph_store
from utils import extract_text


class _ConflictCheck(BaseModel):
    conflicting: bool
    reason: str = ""


def generate_digest(llm: BaseChatModel, period: str = "daily") -> str:
    days = 1 if period == "daily" else 7
    since = (date.today() - timedelta(days=days)).isoformat()

    recent = annotations_db.list_annotations(since=since)
    top_entities = graph_store.most_mentioned_since(since, limit=5)

    if not recent and not top_entities:
        return f"No new activity in the last {days} day(s)."

    lines = ["Recent items:"]
    for item in recent:
        lines.append(f"- [{item['kind']}] {item['text']} (source: {item['source']}, date: {item['date']})")
    if top_entities:
        lines.append("\nMost mentioned:")
        lines.extend(f"- {name} ({count} mentions)" for name, count in top_entities)

    prompt = (
        f"Write a concise {period} digest of the knowledge-base activity below. "
        "Group related items, call out anything that looks decision-worthy or "
        "unresolved.\n\n" + "\n".join(lines)
    )
    response = llm.invoke(prompt)
    return extract_text(response.content)


def repeated_mention_suggestions(threshold: int = 3, days: int = 30) -> list[str]:
    since = (date.today() - timedelta(days=days)).isoformat()
    mentions = graph_store.most_mentioned_since(since, limit=10)
    return [
        f"You mentioned '{name}' {count} times in the last {days} days — want a summary?"
        for name, count in mentions
        if count >= threshold
    ]


def forgotten_open_questions(days: int = 60) -> list[dict]:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    questions = annotations_db.list_annotations(kind="open_question")
    return [q for q in questions if q["status"] == "active" and (q["date"] or "") < cutoff]


def detect_conflicts(llm: BaseChatModel) -> list[dict]:
    decisions = annotations_db.list_annotations(kind="decision")
    by_entity: dict[str, list[dict]] = {}
    for d in decisions:
        if d["entity"]:
            by_entity.setdefault(d["entity"], []).append(d)

    conflicts = []
    for entity, items in by_entity.items():
        if len(items) < 2:
            continue
        newest, prior = sorted(items, key=lambda x: x["date"] or "", reverse=True)[:2]
        try:
            structured_llm = llm.with_structured_output(_ConflictCheck)
            result = structured_llm.invoke(
                f"Do these two statements about '{entity}' conflict with each other?\n"
                f"1. {prior['text']}\n2. {newest['text']}"
            )
            if result.conflicting:
                conflicts.append({"entity": entity, "older": prior, "newer": newest, "reason": result.reason})
        except Exception:
            continue
    return conflicts


def generate_knowledge_report(llm: BaseChatModel, topic: str, context: str) -> str:
    prompt = (
        f"Write a structured knowledge report about '{topic}' using ONLY the context "
        "below. Include sections: Overview, Key Decisions, Timeline, Open Questions, Sources.\n\n"
        f"Context:\n{context}"
    )
    response = llm.invoke(prompt)
    return extract_text(response.content)
