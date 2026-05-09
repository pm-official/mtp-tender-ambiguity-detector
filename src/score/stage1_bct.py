"""Stage 1 — Bidder Commitment Test.

The headline novelty. For every clause flagged as an obligation by the
filter, ask Gemini Flash to role-play a senior contractor and tell us what
they would commit to. If the model cannot fill in a specific quantity,
method, or standard reference, the clause is — by construction — flagged
as ambiguous.

The decision rule is deterministic from the structured output: no
threshold, no learned weights, no logistic regression. The "training-free"
property is the architectural simplification that makes BCT defensible
without any manual annotation.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

from src.llm.gemini_client import LLMCallStats, get_default_client
from src.schemas.bct import (
    CANNOT_DETERMINE,
    CONTRACT_DEFINED,
    BCTOutput,
    Commitment,
    evaluate_flag,
)
from src.schemas.clause import Clause

logger = logging.getLogger(__name__)


# ─── Prompts ────────────────────────────────────────────────────────────────
BCT_SYSTEM_PROMPT = (
    "You are a senior contractor preparing to submit a bid on a construction "
    "tender in India. You cannot bid on hand-waves. For every obligation in "
    "the clause below, identify EXACTLY what you would commit to."
)


BCT_USER_TEMPLATE = """\
Clause text:
<<<{clause_text}>>>

For EACH OBLIGATION in this clause, output one entry. An "obligation" is a
specific duty imposed on a party (often the Contractor): a thing they must
do, a deliverable, a constraint, a deadline. Headings, definitions,
preambles, and pure cross-references are NOT obligations — return an empty
commitments list for those.

Each commitment entry has these fields:

  obligation : a one-sentence paraphrase of what the actor must do.

  quantity   : a specific number with unit if the clause pins one down
               (e.g. "365 days", "100 micrograms per cubic metre",
               "≤ 5 mm", "5% of contract price").
               If the clause does NOT specify a measurable quantity for
               the obligation, return EXACTLY the string "{cannot_determine}".

  method     : a specific procedure, test, or process (e.g. "PM10 measurement
               at site boundary using IS 5182-1 Section 5.2", "weekly
               progress report in the format prescribed in Annex C").
               If the clause does NOT specify how the obligation is
               performed or verified, return EXACTLY "{cannot_determine}".

  standard   : an IS code, IRC, MoRTH, or equivalent reference, with
               version where applicable (e.g. "IS 456:2000 §8").
               If the clause does NOT cite a specific standard, return
               EXACTLY "{cannot_determine}".
               If the contract itself defines the standard (e.g. "as per
               this Contract"), return EXACTLY "{contract_defined}".

  missing_info : a list of specific QUESTIONS a bidder would file with
                 the Authority before bidding, IF AND ONLY IF any of
                 quantity / method / standard is "{cannot_determine}".
                 Each question must point at a concrete gap. Use no more
                 than 4 questions per commitment. If all three of
                 quantity, method, standard are concrete, return [].

Be strict and literal. Do NOT infer values from outside the clause text.
If the clause says "reasonable", that is NOT a quantity. If the clause says
"approved standard", that is NOT a standard reference.
"""


def _build_user_message(clause: Clause) -> str:
    return BCT_USER_TEMPLATE.format(
        clause_text=clause.text.strip(),
        cannot_determine=CANNOT_DETERMINE,
        contract_defined=CONTRACT_DEFINED,
    )


# ─── Public single-clause API ───────────────────────────────────────────────
def run_bct_on_clause(
    clause: Clause,
    *,
    model: str | None = None,
) -> tuple[BCTOutput, LLMCallStats]:
    """Call BCT on one clause. Returns (parsed BCTOutput, telemetry stats)."""
    client = get_default_client()
    parsed, stats = client.generate(
        prompt=_build_user_message(clause),
        response_schema=BCTOutput,
        model=model or client.default_flash,
        temperature=0.0,
        system_instruction=BCT_SYSTEM_PROMPT,
    )

    # Defensive normalisation: trim whitespace and coerce missing string fields
    # to CANNOT_DETERMINE so the decision rule is robust.
    fixed_commitments = []
    for c in parsed.commitments:
        fixed_commitments.append(Commitment(
            obligation=c.obligation.strip(),
            quantity=(c.quantity or CANNOT_DETERMINE).strip(),
            method=(c.method or CANNOT_DETERMINE).strip(),
            standard=(c.standard or CANNOT_DETERMINE).strip(),
            missing_info=[q.strip() for q in c.missing_info if q.strip()],
        ))
    parsed = BCTOutput(commitments=fixed_commitments)
    return parsed, stats


# ─── Iteration helper used by the pipeline ──────────────────────────────────
def run_bct_on_clauses(
    clauses: Iterable[Clause],
    *,
    model: str | None = None,
    progress_every: int = 10,
):
    """Generator that yields (clause, BCTOutput, stats) per clause.

    Use sparingly on large corpora — this is the most expensive single step
    of the pipeline. Throttling is delegated to the underlying Gemini client.
    """
    clauses = list(clauses)
    total = len(clauses)
    for i, clause in enumerate(clauses, 1):
        t0 = time.monotonic()
        try:
            bct, stats = run_bct_on_clause(clause, model=model)
        except Exception as e:
            logger.warning("BCT call failed for %s: %s", clause.clause_id, e)
            yield clause, None, None
            continue
        if i % progress_every == 0 or i == total:
            logger.info("BCT progress: %d/%d  (last cost INR %.4f, %.2fs)",
                        i, total, stats.cost_inr, time.monotonic() - t0)
        yield clause, bct, stats


# Re-export for convenience
__all__ = [
    "run_bct_on_clause",
    "run_bct_on_clauses",
    "BCT_SYSTEM_PROMPT",
    "BCT_USER_TEMPLATE",
    "evaluate_flag",
]
