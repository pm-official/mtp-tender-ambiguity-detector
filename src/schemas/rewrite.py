"""Pydantic schemas for Stage 4 (IS-code-grounded rewrite).

Stage 4 takes each CONFIRMED REAL flag (Stage 3 output) and produces a
rewrite candidate grounded in cited IS-code / CPWD sections. Three
guardrails ensure rewrites are safe:

  G1 Citation integrity — every cited (code_id, version, section) must
     resolve in the local IS-code corpus.
  G2 Re-feed — the rewrite is passed back through Stage 1 BCT; if any of
     {quantity, method, standard} comes back CANNOT_DETERMINE, reject.
  G3 Intent preservation — embedding cosine similarity rewrite vs original
     must be ≥ 0.6; otherwise the rewrite has changed contractual scope.

Up to 3 retry attempts with progressively stricter prompts are permitted.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ─── IS-code corpus chunk ──────────────────────────────────────────────────
class ISCodeChunk(BaseModel):
    """One section-level chunk of an IS / CPWD code, ready for retrieval."""

    chunk_id: str                       # "IS_456_2000::s::5.4.2"
    code_id: str                        # "IS 456:2000"
    code_title: str                     # "Plain and Reinforced Concrete"
    version_year: Optional[int] = None  # 2000
    section: str                        # "5.4.2"
    section_path: list[str] = Field(default_factory=list)
    page: int = 0
    text: str
    char_count: int = 0
    source_file: str = ""               # filename stem
    notes: str = ""


# ─── Citation parsed from a rewrite ────────────────────────────────────────
class ISCodeCitation(BaseModel):
    """A single (code, version, section) reference in a rewrite."""

    code: str                            # e.g. "IS 456" — without version
    version: Optional[str] = None        # e.g. "2000"
    section: Optional[str] = None        # e.g. "5.4.2"
    raw_text: str = ""                   # the literal substring in the rewrite
    resolved: bool = False
    resolved_chunk_id: Optional[str] = None
    rejection_reason: str = ""


# ─── Rewrite output ────────────────────────────────────────────────────────
RewriteStatus = Literal["ACCEPTED", "REJECTED", "PASSED_GUARDRAILS_FAILED",
                         "RETRY_LIMIT_REACHED", "ERROR"]


class Rewrite(BaseModel):
    """Per-flag final rewrite record."""

    flag_id: str
    clause_id: str
    tender_id: str
    doc_id: str

    original_text: str

    # The rewrite proposal
    rewrite_text: str = ""
    is_code_citations: list[ISCodeCitation] = Field(default_factory=list)
    rewrite_explanation: str = ""

    # Guardrail outcomes
    citation_integrity_passed: Optional[bool] = None
    citation_resolved_count: int = 0
    citation_unresolved_count: int = 0

    reflag_check_passed: Optional[bool] = None
    reflag_check_reason: str = ""
    # Differential G2: count of CANNOT_DETERMINE fields in original BCT vs
    # rewrite BCT, summed across {quantity, method, standard} for every
    # extracted commitment. G2 passes if rewrite_cd_count <= original_cd_count
    # AND at least one IS-code citation resolves (G1). This is a non-regression
    # test on bidder uncertainty rather than a perfect-spec test, which is the
    # right abstraction since Stage 1 BCT's obligation decomposition can vary
    # between original and rewrite (e.g. extracting validity period as its own
    # sub-obligation).
    original_cd_count: int = 0
    rewrite_cd_count: int = 0

    intent_cosine: float = 0.0
    intent_check_passed: Optional[bool] = None

    # Final outcome
    status: RewriteStatus = "ERROR"
    rejection_reason: str = ""
    attempts: int = 0                     # 1..max_attempts

    # Telemetry
    cost_inr: float = 0.0
    seconds_elapsed: float = 0.0
    error: Optional[str] = None
