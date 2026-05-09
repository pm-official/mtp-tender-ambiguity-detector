"""Obligation filter — Step 1 of Stage 1 (BCT).

Not every clause imposes an obligation: tables of contents, headings,
preambles, and definitions don't commit anyone to anything. We filter
those out cheaply (one batched Gemini Flash call per ~50 clauses) so the
expensive BCT call only sees clauses that actually need it.

Output: a dict {clause_id -> ObligationCheck} for every clause in the input.
"""

from __future__ import annotations

import logging
from typing import Iterable

from src.llm.gemini_client import get_default_client
from src.schemas.bct import ObligationCheck, ObligationCheckBatch
from src.schemas.clause import Clause

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50
MAX_CLAUSE_CHARS_FOR_FILTER = 600   # truncate very long clauses to keep tokens predictable


def _build_prompt(batch: list[Clause]) -> str:
    """Build the obligation-classification prompt for a batch of clauses."""
    lines = [
        "You decide which clauses from a construction tender impose an actionable",
        "OBLIGATION on a contracting party. An obligation has an actor (Contractor,",
        "Engineer, Authority...) AND an action or duty (must do X, shall provide Y,",
        "is responsible for Z).",
        "",
        "Non-obligations include: headings, table-of-contents entries, preamble or",
        "scope-statement text, pure definitions ('X means Y'), table cells with",
        "captions only, page numbers, copyright notices, and bare cross-references",
        "('See Annex C').",
        "",
        "For EACH clause below, return is_obligation true or false with a brief",
        "rationale (≤ 12 words).",
        "",
        "Clauses:",
    ]
    for c in batch:
        text = c.text.strip().replace("\n", " ")
        if len(text) > MAX_CLAUSE_CHARS_FOR_FILTER:
            text = text[:MAX_CLAUSE_CHARS_FOR_FILTER] + "..."
        lines.append(f"  [{c.clause_id}] {text}")
    lines.append("")
    lines.append("Output strict JSON: { 'results': [ {clause_id, is_obligation, rationale}, ... ] }")
    return "\n".join(lines)


def filter_obligations(
    clauses: Iterable[Clause],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str | None = None,
) -> dict[str, ObligationCheck]:
    """Run the batched obligation filter. Returns {clause_id -> ObligationCheck}.

    Clauses absent from the LLM response (parse failure, batch retry exhausted)
    default to is_obligation=True (conservative — better to over-call BCT
    than to silently skip a real obligation).
    """
    clauses = list(clauses)
    if not clauses:
        return {}

    client = get_default_client()
    out: dict[str, ObligationCheck] = {}

    for i in range(0, len(clauses), batch_size):
        batch = clauses[i : i + batch_size]
        prompt = _build_prompt(batch)
        try:
            reply, stats = client.generate(
                prompt=prompt,
                response_schema=ObligationCheckBatch,
                model=model or client.default_flash,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(
                "Obligation-filter batch %d-%d failed (%s); defaulting all to is_obligation=True",
                i, i + len(batch), e,
            )
            for c in batch:
                out[c.clause_id] = ObligationCheck(
                    clause_id=c.clause_id,
                    is_obligation=True,
                    rationale="filter call failed; defaulting to obligation",
                )
            continue

        # Map LLM responses back to clauses
        seen_ids = set()
        for r in reply.results:
            seen_ids.add(r.clause_id)
            out[r.clause_id] = r
        # Default any clauses the LLM didn't return for
        for c in batch:
            if c.clause_id not in seen_ids:
                out[c.clause_id] = ObligationCheck(
                    clause_id=c.clause_id,
                    is_obligation=True,
                    rationale="not in LLM response; defaulting to obligation",
                )

    return out
