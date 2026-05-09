"""Pydantic schemas for the structural extractions produced by Session 4.

Definition  — a defined-term entry mined from a Definitions section or
              from "X means Y" patterns scattered through the tender.
Quantity    — a numerical commitment extracted via regex, with a canonical
              referent label produced by Gemini Flash.
Reference   — a directed edge in the cross-reference graph (clause → target).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ─── Definition ─────────────────────────────────────────────────────────────
class Definition(BaseModel):
    """A defined term mined from the tender."""

    canonical_form: str                     # e.g. "Contractor", "Engineer"
    definition_text: str                    # the full RHS of "X means Y"
    defined_in_clause: str                  # source clause_id
    doc_id: str
    tender_id: str
    extraction_method: Literal["regex", "llm"]
    aliases: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    notes: str = ""


# ─── Quantity ───────────────────────────────────────────────────────────────
QUANTITY_KINDS = (
    "duration",         # days/weeks/months/years
    "money",            # INR/Rs/USD/lakh/crore
    "dimension",        # mm/cm/m/in/ft
    "percentage",       # %
    "ratio",            # 1:N
    "tolerance",        # ±X
    "count",            # plain integer count
    "frequency",        # per day/per week
    "other",
)


class Quantity(BaseModel):
    """A numerical commitment found in tender text."""

    quantity_id: str                        # "<doc_id>::q::<clause_index>::<n>"
    clause_id: str
    doc_id: str
    tender_id: str

    raw_text: str                           # the matched substring
    surrounding_text: str = ""              # ~80-token window (for referent classification)

    kind: str = "other"                     # one of QUANTITY_KINDS (kept as str for forward-compat)
    value: float = 0.0                      # raw numeric value
    unit: str = ""                          # raw unit token

    value_norm: float = 0.0                 # canonical-unit value
    unit_norm: str = ""                     # canonical unit (days / inr_lakhs / mm / pct / ratio / tolerance / count / per_day)

    # Referent classification (LLM-assigned)
    referent: str = "UNCLASSIFIED"          # canonical referent label
    referent_confidence: float = 0.0
    referent_method: Literal["regex_hint", "llm", "manual", "unclassified"] = "unclassified"

    page: int = 0                           # carried from clause for convenience
    notes: str = ""


# ─── Reference ──────────────────────────────────────────────────────────────
class Reference(BaseModel):
    """A directed edge from one clause to a target — a clause/annex/standard."""

    from_clause: str                        # source clause_id
    target_kind: Literal[
        "clause", "section", "annex", "appendix", "schedule",
        "standard", "drawing", "external", "unknown",
    ]
    target_label: str                       # human-readable target ("Clause 4.6", "Annex C", "IS 456:2000 §12")
    target_id: Optional[str] = None         # resolved clause_id if internal and resolvable
    raw_text: str                           # the literal mention in the source clause
    confidence: float = 1.0
    resolved: bool = False
    notes: str = ""
