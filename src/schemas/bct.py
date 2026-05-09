"""Pydantic schemas for the Bidder Commitment Test (Stage 1).

The headline novelty of the thesis. Reframes ambiguity from a classification
problem ("is this clause ambiguous?") to a commitment elicitation problem
("what would a bidder commit to from this clause?"). The LLM's inability to
fill a specific quantity, method, or standard reference becomes the
deterministic ambiguity signal.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# Sentinel strings the LLM is instructed to use when it cannot pin down a
# specific commitment. Any of these in {quantity, method, standard} = flag.
CANNOT_DETERMINE = "CANNOT_DETERMINE"
CONTRACT_DEFINED = "CONTRACT_DEFINED"

# A clause is flagged iff any of these is true for any of its commitments.
# Listed here so the rule is discoverable from one place.
SENTINEL_FLAG_VALUES = {CANNOT_DETERMINE}


# ─── Step 1: obligation filter ─────────────────────────────────────────────
class ObligationCheck(BaseModel):
    """Per-clause "is this an obligation?" verdict."""

    clause_id: str
    is_obligation: bool
    rationale: str = ""


class ObligationCheckBatch(BaseModel):
    """Batched response for the obligation filter prompt."""

    results: list[ObligationCheck] = Field(default_factory=list)


# ─── Step 2: BCT itself ────────────────────────────────────────────────────
class Commitment(BaseModel):
    """One bidder-side commitment elicited from a single obligation."""

    obligation: str
    quantity: str = CANNOT_DETERMINE
    method: str = CANNOT_DETERMINE
    standard: str = CANNOT_DETERMINE
    missing_info: list[str] = Field(default_factory=list)


class BCTOutput(BaseModel):
    """Full BCT output for one clause (may contain 0+ commitments)."""

    commitments: list[Commitment] = Field(default_factory=list)


# ─── Step 3: the flag ──────────────────────────────────────────────────────
class CandidateFlag(BaseModel):
    """A clause that BCT has labelled as ambiguous (or surveyed but not flagged)."""

    flag_id: str                      # "<doc_id>::flag::<clause_id_short>"
    clause_id: str
    doc_id: str
    tender_id: str
    page: int

    # The original clause text — denormalised here so the audit log is
    # self-contained. Long clauses are kept full; downstream stages reference
    # by clause_id when they need to.
    clause_text: str = ""
    section_path: list[str] = Field(default_factory=list)
    clause_number: Optional[str] = None

    # BCT output and decision
    bct_output: BCTOutput
    flagged: bool
    flag_reason: str = ""             # human-readable summary

    # Provenance: was BCT skipped for this clause (e.g. obligation filter said no)?
    is_obligation: bool = True
    skipped_reason: Optional[str] = None

    # Cost / latency telemetry
    cost_inr: float = 0.0
    seconds_elapsed: float = 0.0


# ─── Decision rule ─────────────────────────────────────────────────────────
def evaluate_flag(bct_output: BCTOutput) -> tuple[bool, str]:
    """Apply the deterministic decision rule. Returns (flagged, reason)."""
    if not bct_output.commitments:
        return False, "no commitments extracted (clause has no actionable obligation)"

    triggers: list[str] = []
    for c in bct_output.commitments:
        gaps = []
        if c.quantity == CANNOT_DETERMINE:
            gaps.append("quantity")
        if c.method == CANNOT_DETERMINE:
            gaps.append("method")
        if c.standard == CANNOT_DETERMINE:
            gaps.append("standard")
        if c.missing_info:
            gaps.append(f"missing_info({len(c.missing_info)} questions)")
        if gaps:
            triggers.append(f"obligation '{c.obligation[:60]}': {', '.join(gaps)}")

    if not triggers:
        return False, "all commitments specify quantity, method, and standard"

    reason = "; ".join(triggers[:3])
    if len(triggers) > 3:
        reason += f"; ... (+{len(triggers) - 3} more)"
    return True, reason
