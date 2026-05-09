"""LLM fallback for Definitions extraction (Session 4 helper).

Sent clauses get batched (10 per call) to keep cost low. Each call asks
Gemini Flash to return any (term, definition) pairs it spots.
"""

from __future__ import annotations

import logging
from typing import Iterable

from pydantic import BaseModel, Field

from src.llm.gemini_client import get_default_client
from src.schemas.clause import Clause
from src.schemas.structures import Definition

logger = logging.getLogger(__name__)

BATCH_SIZE = 10


class _Pair(BaseModel):
    clause_id: str
    term: str
    definition: str


class _Reply(BaseModel):
    pairs: list[_Pair] = Field(default_factory=list)


def llm_extract_definitions(clauses: Iterable[Clause]) -> list[Definition]:
    clauses = list(clauses)
    if not clauses:
        return []

    client = get_default_client()
    out: list[Definition] = []

    for i in range(0, len(clauses), BATCH_SIZE):
        batch = clauses[i : i + BATCH_SIZE]
        prompt = _build_prompt(batch)
        try:
            reply, stats = client.generate(
                prompt=prompt,
                response_schema=_Reply,
                model=client.default_flash,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("LLM definitions batch failed (skipping): %s", e)
            continue

        for pair in reply.pairs:
            term = pair.term.strip()
            body = pair.definition.strip()
            if not term or not body or len(body) < 10:
                continue
            cl = next((c for c in batch if c.clause_id == pair.clause_id), None)
            if cl is None:
                continue
            out.append(Definition(
                canonical_form=term,
                definition_text=body,
                defined_in_clause=cl.clause_id,
                doc_id=cl.doc_id,
                tender_id=cl.tender_id,
                extraction_method="llm",
                confidence=0.7,
            ))
    return out


def _build_prompt(batch: list[Clause]) -> str:
    lines = [
        "You extract DEFINED TERMS from construction-tender clauses.",
        "",
        "For each clause below, list every (term, definition) pair where the",
        "clause says 'X means Y', 'X shall mean Y', 'X is defined as Y',",
        "or any equivalent definition pattern. Return strict JSON.",
        "",
        "Be strict: do NOT extract general statements as definitions. Only",
        "actual term-definitions count. If a clause has no definitions,",
        "do not return an entry for it.",
        "",
        "Clauses:",
    ]
    for c in batch:
        lines.append(f"  [{c.clause_id}] {c.text[:1500]}")
    lines.append("")
    lines.append("Output JSON: { 'pairs': [ {clause_id, term, definition}, ... ] }")
    return "\n".join(lines)
