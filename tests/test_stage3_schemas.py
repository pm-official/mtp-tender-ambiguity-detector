"""Offline tests for Stage 3 schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.stage3 import (
    ConfirmedFlag,
    CritiqueVerdict,
    SeverityScore,
    severity_tier,
)


# ─── SeverityScore validation ──────────────────────────────────────────────
def test_severity_score_within_range():
    s = SeverityScore(commercial_exposure=3, dispute_likelihood=4, reviewer_cost=2)
    assert s.commercial_exposure == 3
    assert s.composite_severity == 7


def test_severity_score_rejects_out_of_range():
    with pytest.raises(ValidationError):
        SeverityScore(commercial_exposure=6, dispute_likelihood=4, reviewer_cost=2)
    with pytest.raises(ValidationError):
        SeverityScore(commercial_exposure=0, dispute_likelihood=4, reviewer_cost=2)
    with pytest.raises(ValidationError):
        SeverityScore(commercial_exposure=3, dispute_likelihood=4, reviewer_cost=10)


def test_severity_composite_includes_only_two_dimensions():
    """reviewer_cost must NOT be added to composite (it's an operational metric)."""
    s = SeverityScore(commercial_exposure=5, dispute_likelihood=5, reviewer_cost=5)
    assert s.composite_severity == 10
    s2 = SeverityScore(commercial_exposure=1, dispute_likelihood=1, reviewer_cost=5)
    assert s2.composite_severity == 2


# ─── severity_tier mapping ─────────────────────────────────────────────────
@pytest.mark.parametrize("composite,expected", [
    (10, "high"),
    (8, "high"),
    (7, "medium"),
    (5, "medium"),
    (4, "low"),
    (2, "low"),
    (1, "n/a"),  # below the [2, 10] valid range
])
def test_severity_tier_mapping(composite, expected):
    assert severity_tier(composite) == expected


# ─── CritiqueVerdict literal enforcement ───────────────────────────────────
def test_critique_verdict_accepts_valid_literals():
    for v in ("CONFIRMED", "REJECTED", "WEAK"):
        cv = CritiqueVerdict(verdict=v)
        assert cv.verdict == v


def test_critique_verdict_rejects_unknown_literal():
    with pytest.raises(ValidationError):
        CritiqueVerdict(verdict="UNKNOWN")


# ─── ConfirmedFlag round-trip ──────────────────────────────────────────────
def test_confirmed_flag_round_trip():
    cf = ConfirmedFlag(
        flag_id="f1",
        clause_id="c1",
        doc_id="d.pdf",
        tender_id="T1",
        page=5,
        clause_text="...",
        stage2_verdict="REAL",
        stage2_rationale="bidder cannot commit",
        critique_verdict="CONFIRMED",
        critique_refutation="genuine vagueness",
        severity=SeverityScore(commercial_exposure=4, dispute_likelihood=4, reviewer_cost=3),
        composite_severity=8,
        severity_tier="high",
    )
    payload = cf.model_dump_json()
    restored = ConfirmedFlag.model_validate_json(payload)
    assert restored.severity.composite_severity == 8
    assert restored.severity_tier == "high"


def test_confirmed_flag_default_no_severity():
    """ConfirmedFlag with no severity (e.g. Stage 3 didn't run severity scoring)."""
    cf = ConfirmedFlag(
        flag_id="f1",
        clause_id="c1",
        doc_id="d.pdf",
        tender_id="T1",
        page=1,
    )
    assert cf.severity is None
    assert cf.composite_severity == 0
    assert cf.severity_tier == ""
