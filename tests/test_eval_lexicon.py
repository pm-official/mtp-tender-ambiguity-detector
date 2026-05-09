"""Tests for the rule-based lexicon baseline."""

from __future__ import annotations

import pytest

from src.eval.lexicon_baseline import detect_markers, is_flagged, LEXICON


@pytest.mark.parametrize("text,expected_categories", [
    ("The Contractor shall make best efforts to comply.", ["effort_pledges"]),
    ("Materials shall be of reasonable quality.", ["subjectivity"]),
    ("Reports shall be furnished as directed by the Engineer.", ["delegated_authority"]),
    ("Approximately 100 cubic metres of fill material.", ["approximation"]),
    ("Local labour shall be used wherever feasible.", ["feasibility_dodge"]),
    ("Use other relevant standards as required.",
     ["delegated_authority", "feasibility_dodge", "underspecified_quantifier"]),
])
def test_detect_markers_categories(text, expected_categories):
    matches = detect_markers(text)
    seen = sorted({c for c, _ in matches})
    for cat in expected_categories:
        assert cat in seen, f"Expected category {cat!r} not found in {seen!r} for {text!r}"


def test_detect_markers_returns_empty_for_specific_text():
    """Specific, IS-code-grounded text should not match any markers."""
    text = (
        "The Contractor shall furnish a Bank Guarantee Bond per CPWD GCC "
        "Section 104, equal to 5% of the Contract Price, valid until 90 days "
        "after the end of the Defect Liability Period."
    )
    matches = detect_markers(text)
    assert matches == [], f"Specific text matched markers: {matches}"


def test_is_flagged_threshold():
    text = "Contractor shall make best efforts and use reasonable means."
    # Two markers (best efforts, reasonable)
    assert is_flagged(text, min_markers=1) is True
    assert is_flagged(text, min_markers=2) is True
    assert is_flagged(text, min_markers=3) is False


def test_lexicon_categories_non_empty():
    """Sanity: every category has at least one pattern."""
    assert all(len(v) > 0 for v in LEXICON.values())
    assert len(LEXICON) >= 5
