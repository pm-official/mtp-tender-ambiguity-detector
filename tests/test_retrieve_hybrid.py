"""Offline tests for hybrid retrieval: R2, R3, and RRF fusion."""

from __future__ import annotations

from src.retrieve.hybrid import (
    reciprocal_rank_fusion,
    retrieve_shared_defined_terms,
    retrieve_shared_referents,
)
from src.schemas.clause import Clause
from src.schemas.structures import Definition, Quantity


def _clause(text: str, *, clause_id="c1", page=1, doc="d.pdf") -> Clause:
    return Clause(
        clause_id=clause_id,
        doc_id=doc,
        tender_id="T1",
        page=page,
        page_end=page,
        section_path=[],
        clause_number=None,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
    )


def _defn(term: str, *, doc="d.pdf") -> Definition:
    return Definition(
        canonical_form=term,
        definition_text=f"definition of {term}",
        defined_in_clause=f"{doc}::00001",
        doc_id=doc,
        tender_id="T1",
        extraction_method="regex",
    )


def _qty(clause_id: str, referent: str, *, value=1.0) -> Quantity:
    return Quantity(
        quantity_id=f"{clause_id}::q000",
        clause_id=clause_id,
        doc_id="d.pdf",
        tender_id="T1",
        raw_text="1 day",
        kind="duration",
        value=value,
        unit="day",
        value_norm=value,
        unit_norm="days",
        referent=referent,
        page=1,
    )


# ─── R2 — shared defined-term ──────────────────────────────────────────────
def test_r2_returns_clauses_sharing_defined_term():
    flagged_text = "The Contractor shall maintain the Works under this Contract."
    clauses = [
        _clause("Works are defined as the Permanent Works.", clause_id="cA"),
        _clause("The Contractor shall maintain workmanship.", clause_id="cB"),
        _clause("Unrelated clause about lighting.", clause_id="cC"),
    ]
    defs = [_defn("Works"), _defn("Contractor")]
    out = retrieve_shared_defined_terms(flagged_text, definitions=defs, clauses=clauses)
    ids = [r[0] for r in out]
    assert "cA" in ids   # mentions "Works"
    assert "cB" in ids   # mentions "Contractor"
    assert "cC" not in ids


def test_r2_empty_when_no_overlap():
    out = retrieve_shared_defined_terms(
        "an entirely unrelated clause body", definitions=[_defn("Works")],
        clauses=[_clause("nothing in common", clause_id="cX")],
    )
    assert out == []


def test_r2_handles_no_definitions():
    out = retrieve_shared_defined_terms("any text", definitions=[], clauses=[])
    assert out == []


# ─── R3 — shared canonical referent ────────────────────────────────────────
def test_r3_returns_clauses_with_same_referent():
    flagged = _clause("flagged clause body", clause_id="flagged")
    clauses = [
        flagged,
        _clause("clause about defect liability period", clause_id="cA"),
        _clause("unrelated clause", clause_id="cB"),
    ]
    quantities = [
        _qty("flagged", "defect_liability_period"),
        _qty("cA", "defect_liability_period"),
        _qty("cB", "completion_period"),
    ]
    out = retrieve_shared_referents(flagged, quantities=quantities, clauses=clauses)
    ids = [r[0] for r in out]
    assert "cA" in ids
    assert "cB" not in ids
    assert "flagged" not in ids


def test_r3_empty_when_flagged_has_no_quantities():
    flagged = _clause("flagged with no quantities", clause_id="flagged")
    out = retrieve_shared_referents(
        flagged,
        quantities=[_qty("cA", "completion_period")],
        clauses=[flagged, _clause("a", clause_id="cA")],
    )
    assert out == []


def test_r3_excludes_unclassified_referents():
    flagged = _clause("flagged", clause_id="flagged")
    qs = [
        _qty("flagged", "UNCLASSIFIED"),
        _qty("cA", "UNCLASSIFIED"),
    ]
    out = retrieve_shared_referents(flagged, quantities=qs,
                                     clauses=[flagged, _clause("a", clause_id="cA")])
    assert out == []


# ─── RRF fusion ────────────────────────────────────────────────────────────
def test_rrf_clause_in_multiple_routes_ranks_higher():
    # Same clause in 2 routes vs only in 1
    route_results = {
        "R1": [("c_top", 0.9, "text top", {}), ("c_other", 0.5, "text other", {})],
        "R2": [("c_top", 1.0, "text top", {}), ("c_only_r2", 0.6, "text r2", {})],
    }
    fused = reciprocal_rank_fusion(route_results, k=60, top_k=5)
    # c_top should be #1 because it was found by both routes at high rank
    assert fused[0].clause_id == "c_top"
    assert "R1" in fused[0].sources
    assert "R2" in fused[0].sources
    assert fused[0].rrf_score > fused[1].rrf_score


def test_rrf_returns_at_most_top_k():
    route_results = {
        "R1": [(f"c{i}", 0.5, f"t{i}", {}) for i in range(20)],
    }
    fused = reciprocal_rank_fusion(route_results, top_k=5)
    assert len(fused) == 5


def test_rrf_assigns_sequential_ranks():
    route_results = {
        "R1": [("a", 0.9, "ta", {}), ("b", 0.8, "tb", {}), ("c", 0.7, "tc", {})],
    }
    fused = reciprocal_rank_fusion(route_results, top_k=5)
    assert [h.rank for h in fused] == [1, 2, 3]


def test_rrf_handles_empty_routes():
    fused = reciprocal_rank_fusion({}, top_k=5)
    assert fused == []
    fused2 = reciprocal_rank_fusion({"R1": []}, top_k=5)
    assert fused2 == []


def test_rrf_score_formula_matches_definition():
    """RRF score = sum over routes of 1/(k + rank). For a clause at rank=1
    in two routes with k=60, score should be 2/61."""
    route_results = {
        "R1": [("c", 0.9, "t", {})],
        "R2": [("c", 0.8, "t", {})],
    }
    fused = reciprocal_rank_fusion(route_results, k=60, top_k=5)
    assert len(fused) == 1
    expected = 2.0 / 61
    assert abs(fused[0].rrf_score - expected) < 1e-5
