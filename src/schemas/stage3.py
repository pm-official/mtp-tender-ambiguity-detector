"""Pydantic schemas for Stage 3: critique pass + severity scoring.

Stage 3 takes Stage 2's REAL verdicts and:
  1. Sends each through a second LLM (the "skeptical reviewer") which tries
     to refute the REAL verdict. Surviving REALs become CONFIRMED;
     refuted REALs are demoted to WEAK or rejected outright.
  2. Scores each CONFIRMED flag on three 1-5 dimensions: commercial
     exposure, dispute likelihood, reviewer cost (a separate metric, not
     summed into severity). Composite severity = commercial + dispute,
     range [2, 10].

Per methodology Part 7 — Section 7.1 (critique pass) and 7.2 (severity).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ─── Critique pass ─────────────────────────────────────────────────────────
CritiqueVerdictLiteral = Literal["CONFIRMED", "REJECTED", "WEAK"]


class CritiqueVerdict(BaseModel):
    """Output of the critique pass — one record per Stage-2 REAL verdict."""

    flag_id: str = ""           # patched in by the runner if LLM omits it
    clause_id: str = ""
    verdict: CritiqueVerdictLiteral
    refutation: str = ""        # the skeptical reviewer's reasoning
    resolving_clause_id: Optional[str] = None
    confidence: float = 0.0


# ─── Severity scoring ──────────────────────────────────────────────────────
class SeverityScore(BaseModel):
    """Three-dimensional severity scoring per the Koc & Gurgun (2022) framing.

    All three are integers in [1, 5]. Composite severity (a downstream
    derived field) = commercial_exposure + dispute_likelihood, range [2,10].
    reviewer_cost is reported separately — it is an OPERATIONAL metric, not
    part of the severity score.
    """

    commercial_exposure: int = Field(..., ge=1, le=5)
    dispute_likelihood: int = Field(..., ge=1, le=5)
    reviewer_cost: int = Field(..., ge=1, le=5)
    rationale: str = ""

    @property
    def composite_severity(self) -> int:
        return int(self.commercial_exposure + self.dispute_likelihood)


# ─── Confirmed flag (final Stage-3 record per flag) ────────────────────────
class ConfirmedFlag(BaseModel):
    """Per-flag final record after Stage 3."""

    flag_id: str
    clause_id: str
    doc_id: str
    tender_id: str
    page: int

    # Stage-1 carry-through
    clause_text: str = ""
    section_path: list[str] = Field(default_factory=list)
    clause_number: Optional[str] = None

    # Stage-2 verdict (carried in for audit)
    stage2_verdict: Literal["REAL", "APPARENT", "WEAK"] = "REAL"
    stage2_rationale: str = ""
    resolving_clause_id: Optional[str] = None

    # Stage-3 critique outcome
    critique_verdict: Optional[CritiqueVerdictLiteral] = None  # None if Stage-2 wasn't REAL
    critique_refutation: str = ""

    # Severity (only present for CONFIRMED records)
    severity: Optional[SeverityScore] = None
    composite_severity: int = 0
    severity_tier: str = ""    # "high" / "medium" / "low" / "n/a"

    # Telemetry
    cost_inr: float = 0.0
    seconds_elapsed: float = 0.0
    error: Optional[str] = None


# ─── Severity tiering helper ───────────────────────────────────────────────
def severity_tier(composite: int) -> str:
    """Map composite severity 2..10 to a tier label."""
    if composite >= 8:
        return "high"
    if composite >= 5:
        return "medium"
    if composite >= 2:
        return "low"
    return "n/a"
