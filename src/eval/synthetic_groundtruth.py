"""Hand-labelled ground truth for the synthetic SYN_001 tender.

SYN_001 was generated (in src/acquire/synthetic.py) with a deliberate mix of
vague (Type F) and precise clauses. Because we wrote it ourselves, we know
exactly which clauses ARE vague-by-design — these are the "silver" labels
this module exposes.

Two label sets are provided:

  SYN_001_VAGUE_CLAUSE_IDS  — clause_ids that are vague-by-design.
                               Use these as the positive ground truth set.
  SYN_001_PRECISE_CLAUSE_IDS — clause_ids that are precise-by-design.
                               Used to compute true negatives.

Anything outside both sets (e.g. document-header lines, BoQ blob) is left
out of the evaluation universe entirely — we cannot make a confident
judgement at the clause level.

Each vague clause is also tagged with the failure mode it exhibits, so
per-category precision/recall can be computed for finer ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GroundTruthLabel:
    """A single labelled clause."""

    clause_id: str
    vague: bool
    category: str  # "subjectivity" / "delegated_authority" / etc., or "precise"
    note: str = ""


# Categories follow the taxonomy used by the lexicon baseline so per-category
# precision/recall comparisons are apples-to-apples:
#   subjectivity        — "reasonable", "adequate", "suitable"
#   effort_pledges      — "best efforts", "due diligence"
#   delegated_authority — "as directed", "acceptable to the Authority"
#   feasibility_dodge   — "wherever feasible", "to the extent practicable"
#   underspecified_quantifier — "sufficient frequency", "regular intervals"
#   approximation       — "approximately", "around"
#
# A clause may exhibit MULTIPLE failure modes; we tag the dominant one.

SYN_001_LABELS: tuple[GroundTruthLabel, ...] = (
    # ─── NIT (Notice Inviting Tender) ──────────────────────────────────────
    GroundTruthLabel("01_nit.pdf::00001", vague=False, category="precise",
                     note="Scope cites MoRTH 5.0 specifically."),
    GroundTruthLabel("01_nit.pdf::00002", vague=False, category="precise",
                     note="Estimated cost INR 250 crore — specific amount."),
    GroundTruthLabel("01_nit.pdf::00003", vague=False, category="precise",
                     note="Bid submission date/time/portal all specific."),
    GroundTruthLabel("01_nit.pdf::00004", vague=True, category="delegated_authority",
                     note="'A suitable date which will be notified separately'."),
    GroundTruthLabel("01_nit.pdf::00005", vague=True, category="subjectivity",
                     note="'adequate financial capacity', 'relevant past experience'."),

    # ─── GCC (General Conditions of Contract) ──────────────────────────────
    GroundTruthLabel("02_gcc.pdf::00001", vague=False, category="precise",
                     note="Definitions; binding by reference."),
    GroundTruthLabel("02_gcc.pdf::00002", vague=False, category="precise",
                     note="Commencement: 15 days after LoA, 365 days to completion."),
    GroundTruthLabel("02_gcc.pdf::00003", vague=True, category="subjectivity",
                     note="'workmanlike manner', 'best industry practices', "
                          "'satisfaction of the Engineer'."),
    GroundTruthLabel("02_gcc.pdf::00004", vague=True, category="subjectivity",
                     note="'suitable quality', 'approved sources', 'relevant Indian Standard'."),
    GroundTruthLabel("02_gcc.pdf::00005", vague=False, category="precise",
                     note="DLP = 12 months from Completion Certificate."),
    GroundTruthLabel("02_gcc.pdf::00006", vague=True, category="subjectivity",
                     note="'reasonably necessary', 'fair compensation'."),
    GroundTruthLabel("02_gcc.pdf::00007", vague=False, category="precise",
                     note="Termination defers to Clause 18; 30 days notice for convenience."),

    # ─── SCC (Special Conditions of Contract) — heavily vague v1 ──────────
    GroundTruthLabel("03_scc.pdf::00001", vague=True, category="delegated_authority",
                     note="'in a form acceptable to the Authority'."),
    GroundTruthLabel("03_scc.pdf::00002", vague=True, category="subjectivity",
                     note="'adequate insurance covering all risks'."),
    GroundTruthLabel("03_scc.pdf::00003", vague=True, category="subjectivity",
                     note="'reasonable levels of dust suppression', 'appropriate measures'. "
                          "(Amended later by Corrigendum 1 — but v1 text is still vague.)"),
    GroundTruthLabel("03_scc.pdf::00004", vague=True, category="underspecified_quantifier",
                     note="'sufficient number of qualified safety officers', "
                          "'appropriately trained'."),
    GroundTruthLabel("03_scc.pdf::00005", vague=True, category="underspecified_quantifier",
                     note="'progress reports at regular intervals'. (Amended later "
                          "by Corrigendum 2 — but v1 text is still vague.)"),
    GroundTruthLabel("03_scc.pdf::00006", vague=True, category="underspecified_quantifier",
                     note="'relevant IS code', 'sufficient frequency'."),
    GroundTruthLabel("03_scc.pdf::00007", vague=True, category="effort_pledges",
                     note="'best efforts to employ local labour wherever feasible'."),

    # ─── Specs (heavily precise; one vague example) ─────────────────────────
    GroundTruthLabel("04_specs.pdf::00001", vague=False, category="precise",
                     note="M30 concrete, IS 456:2000 §9 — fully specified."),
    GroundTruthLabel("04_specs.pdf::00002", vague=False, category="precise",
                     note="Fe 500D bars per IS 1786:2008."),
    GroundTruthLabel("04_specs.pdf::00003", vague=False, category="precise",
                     note="98% MDD per IS 2720 Part 7 at 200 m intervals."),
    GroundTruthLabel("04_specs.pdf::00004", vague=False, category="precise",
                     note="50 mm thickness, ≥145 °C, MoRTH §509."),
    GroundTruthLabel("04_specs.pdf::00005", vague=False, category="precise",
                     note="Drainage geometry fully specified."),
    GroundTruthLabel("04_specs.pdf::00006", vague=False, category="precise",
                     note="Curing 7 days minimum by ponding/wet hessian."),
    GroundTruthLabel("04_specs.pdf::00007", vague=True, category="feasibility_dodge",
                     note="'aesthetically pleasing', 'to the extent practicable'."),
)


# ─── Convenience views ─────────────────────────────────────────────────────
SYN_001_VAGUE_CLAUSE_IDS: frozenset[str] = frozenset(
    g.clause_id for g in SYN_001_LABELS if g.vague
)

SYN_001_PRECISE_CLAUSE_IDS: frozenset[str] = frozenset(
    g.clause_id for g in SYN_001_LABELS if not g.vague
)

SYN_001_UNIVERSE: frozenset[str] = SYN_001_VAGUE_CLAUSE_IDS | SYN_001_PRECISE_CLAUSE_IDS


def label_for(clause_id: str) -> Optional[GroundTruthLabel]:
    """Return the GroundTruthLabel for a clause_id, or None if unlabelled."""
    for g in SYN_001_LABELS:
        if g.clause_id == clause_id:
            return g
    return None


def vague_by_category(category: str) -> frozenset[str]:
    """Return the frozenset of vague clause_ids tagged with `category`."""
    return frozenset(
        g.clause_id for g in SYN_001_LABELS if g.vague and g.category == category
    )


def category_counts() -> dict[str, int]:
    """How many vague clauses exist per category, for the eval summary."""
    out: dict[str, int] = {}
    for g in SYN_001_LABELS:
        if g.vague:
            out[g.category] = out.get(g.category, 0) + 1
    return out


def get_ground_truth(tender_id: str) -> Optional[frozenset[str]]:
    """Return the set of vague clause_ids for a known synthetic tender, or
    None if the tender_id is not recognised."""
    if tender_id == "SYN_001":
        return SYN_001_VAGUE_CLAUSE_IDS
    return None


__all__ = [
    "GroundTruthLabel",
    "SYN_001_LABELS",
    "SYN_001_VAGUE_CLAUSE_IDS",
    "SYN_001_PRECISE_CLAUSE_IDS",
    "SYN_001_UNIVERSE",
    "label_for",
    "vague_by_category",
    "category_counts",
    "get_ground_truth",
]
