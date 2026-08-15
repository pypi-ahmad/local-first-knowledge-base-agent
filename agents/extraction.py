"""LLM-based extraction used during indexing: entities/relations for the
knowledge graph, and decisions/action-items/open-questions for proactive
features. One structured call per chunk (not two) to halve LLM round-trips.

ponytail: no dedicated NER model (spaCy etc.) — reuses the same local LLM
already required for generation. Silently degrades to empty results if the
model doesn't support structured output well; extraction is a bonus layer,
not load-bearing for retrieval.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, field_validator

_MAX_CHARS = 4000


def _normalize_kind(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


class Entity(BaseModel):
    name: str
    kind: str = "entity"

    _normalize = field_validator("kind")(classmethod(lambda cls, v: _normalize_kind(v)))


class Relation(BaseModel):
    source: str
    target: str
    relation: str


class Annotation(BaseModel):
    kind: str = Field(description="one of: decision, action_item, open_question")
    text: str
    entity: str | None = None

    _normalize = field_validator("kind")(classmethod(lambda cls, v: _normalize_kind(v)))


class ChunkExtraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)


_PROMPT = """Extract structured knowledge from this text:
1. entities: named people, projects, organizations, or concepts explicitly mentioned.
2. relations: (source, relation, target) triples between those entities, stated or clearly implied.
3. annotations: decisions made, action items, or open questions explicitly present in the text.

Only extract what is clearly stated. Return empty lists if nothing qualifies.

Text:
{text}"""


def extract_from_chunk(llm: BaseChatModel, text: str) -> ChunkExtraction:
    try:
        structured_llm = llm.with_structured_output(ChunkExtraction)
        result = structured_llm.invoke(_PROMPT.format(text=text[:_MAX_CHARS]))
        return result if isinstance(result, ChunkExtraction) else ChunkExtraction()
    except Exception:
        return ChunkExtraction()
