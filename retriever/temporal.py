"""Resolve relative time expressions ("last month", "in January", "recently")
into an ISO date range, so retrieval can filter on stored document dates.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Optional

from dateparser.search import search_dates

DateRange = tuple[Optional[str], Optional[str]]

_QUARTER_RE = re.compile(r"\bq([1-4])(?:\s+of)?(?:\s+(\d{4}))?\b", re.IGNORECASE)
_BEFORE_AFTER_RE = re.compile(r"\b(before|after|since)\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)")


def _quarter_range(quarter: int, year: int) -> DateRange:
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2
    start = date(year, start_month, 1)
    end_day = calendar.monthrange(year, end_month)[1]
    return start.isoformat(), date(year, end_month, end_day).isoformat()


def _entity_anchored_range(query: str) -> Optional[DateRange]:
    """'before/after/since <Entity>' anchored to that entity's first-seen date
    in the knowledge graph. ponytail: only works for named entities already in
    the graph (e.g. 'Project Atlas'); generic phrases ('the project') aren't
    resolved — the user needs to name the thing."""
    match = _BEFORE_AFTER_RE.search(query)
    if not match:
        return None
    from db import graph_store  # local import: avoid a hard dependency for callers that don't need it

    direction, phrase = match.group(1).lower(), match.group(2).strip()
    candidates = graph_store.search_entities(phrase, limit=1)
    if not candidates:
        return None
    anchor = graph_store.entity_first_seen(candidates[0])
    if not anchor:
        return None
    if direction == "before":
        return None, anchor
    return anchor, None  # "after" / "since"


def parse_date_range(query: str, today: Optional[date] = None) -> Optional[DateRange]:
    today = today or date.today()
    q = query.lower()

    quarter_match = _QUARTER_RE.search(q)
    if quarter_match:
        year = int(quarter_match.group(2)) if quarter_match.group(2) else today.year
        return _quarter_range(int(quarter_match.group(1)), year)

    entity_range = _entity_anchored_range(query)
    if entity_range:
        return entity_range

    if "yesterday" in q:
        d = today - timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if "today" in q:
        return today.isoformat(), today.isoformat()
    if "last month" in q:
        first_this_month = today.replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start.isoformat(), last_month_end.isoformat()
    if "this month" in q:
        return today.replace(day=1).isoformat(), today.isoformat()
    if "last week" in q:
        start_this_week = today - timedelta(days=today.weekday())
        last_week_end = start_this_week - timedelta(days=1)
        last_week_start = last_week_end - timedelta(days=6)
        return last_week_start.isoformat(), last_week_end.isoformat()
    if "this week" in q:
        start_this_week = today - timedelta(days=today.weekday())
        return start_this_week.isoformat(), today.isoformat()
    if "recently" in q or "lately" in q:
        return (today - timedelta(days=14)).isoformat(), today.isoformat()

    return _parse_explicit_mention(query, today)


def _parse_explicit_mention(query: str, today: date) -> Optional[DateRange]:
    """Fall back to dateparser for an explicit mention like 'in January' or '2025-03-14'."""
    settings = {
        "PREFER_DATES_FROM": "past",
        "RELATIVE_BASE": datetime.combine(today, datetime.min.time()),
    }
    found = search_dates(query, settings=settings, languages=["en"])
    if not found:
        return None

    matched_text, parsed = found[0]
    if re.search(r"\d{1,2}(?!\d)", matched_text) is None:
        # No day-of-month in the matched text ("in January") -> whole-month range.
        start = parsed.date().replace(day=1)
        end_day = calendar.monthrange(start.year, start.month)[1]
        return start.isoformat(), start.replace(day=end_day).isoformat()

    d = parsed.date()
    return d.isoformat(), d.isoformat()


def _demo() -> None:
    fixed_today = date(2026, 8, 15)
    assert parse_date_range("what did I decide last month", fixed_today) == ("2026-07-01", "2026-07-31")
    assert parse_date_range("notes from today", fixed_today) == ("2026-08-15", "2026-08-15")
    assert parse_date_range("anything about last week", fixed_today) == ("2026-08-03", "2026-08-09")
    assert parse_date_range("what did I write in January", fixed_today) == ("2026-01-01", "2026-01-31")
    assert parse_date_range("what is the capital of France", fixed_today) is None
    assert parse_date_range("what happened in Q1 2026", fixed_today) == ("2026-01-01", "2026-03-31")

    from db import graph_store

    graph_store.clear_all()
    graph_store.upsert_entity("Project Atlas", "project", "2026-03-01")
    assert parse_date_range("what did we decide before Project Atlas started", fixed_today) == (None, "2026-03-01")
    assert parse_date_range("what happened after Project Atlas kicked off", fixed_today) == ("2026-03-01", None)
    graph_store.clear_all()
    print("retriever.temporal self-check ok")


if __name__ == "__main__":
    _demo()
