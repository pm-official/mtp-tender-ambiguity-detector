"""Offline tests for the IS-code citation parser + verifier."""

from __future__ import annotations

import pytest

from src.iscode.citation import (
    parse_citations,
    verify_citation,
)
from src.schemas.rewrite import ISCodeCitation


# ─── Citation parsing ──────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected_count,expected_codes", [
    ("Concrete shall conform to IS 456:2000 Section 8.1.", 1, ["IS 456"]),
    ("Per IS 1786:2008 §3 and IS 456:2000 §6.4.1.", 2, ["IS 1786", "IS 456"]),
    ("Use IS 1200 (Part 5):1982 Section 3 for measurement.", 1, ["IS 1200 (Part 5)"]),
    ("Conforms to CPWD Specifications:2019 §5.4.", 1, ["CPWD"]),
    ("Just plain text without any citation.", 0, []),
])
def test_parse_citations_counts(text, expected_count, expected_codes):
    cs = parse_citations(text)
    assert len(cs) == expected_count
    for code in expected_codes:
        assert any(code in c.code for c in cs)


def test_parse_citations_extracts_section():
    cs = parse_citations("Per IS 456:2000 Section 8.1.")
    assert len(cs) == 1
    assert cs[0].section == "8.1"
    assert cs[0].version == "2000"


def test_parse_citations_handles_paren_clause():
    cs = parse_citations("(IS 456:2000 Cl. 5.4.1) applies here.")
    assert len(cs) == 1
    assert cs[0].section == "5.4.1"


def test_parse_citations_dedupes():
    """Same citation appearing twice should be deduplicated."""
    cs = parse_citations("IS 456:2000 §8.1 ... IS 456:2000 §8.1 again.")
    assert len(cs) == 1


# ─── Citation verification ─────────────────────────────────────────────────
def test_verify_with_empty_registry():
    """Verification against an empty registry should mark all citations unresolved."""
    c = ISCodeCitation(code="IS 456", version="2000", section="8.1", raw_text="IS 456:2000 §8.1")
    verified = verify_citation(c, registry={})
    assert verified.resolved is False
    assert "registry empty" in verified.rejection_reason


def test_verify_exact_match_resolves():
    registry = {
        "IS 456:2000::8.1": {
            "chunk_id": "IS_456_2000::s::8.1::001",
            "code_id": "IS 456:2000",
            "section": "8.1",
        },
    }
    c = ISCodeCitation(code="IS 456", version="2000", section="8.1")
    verified = verify_citation(c, registry=registry)
    assert verified.resolved is True
    assert verified.resolved_chunk_id == "IS_456_2000::s::8.1::001"


def test_verify_prefix_match_resolves():
    """Citation §8 should resolve if registry has §8.1 (prefix match)."""
    registry = {
        "IS 456:2000::8.1": {"chunk_id": "x", "code_id": "IS 456:2000", "section": "8.1"},
    }
    c = ISCodeCitation(code="IS 456", version="2000", section="8")
    verified = verify_citation(c, registry=registry)
    assert verified.resolved is True


def test_verify_unresolved_returns_reason():
    registry = {"IS 456:2000::8.1": {"chunk_id": "x", "code_id": "IS 456:2000", "section": "8.1"}}
    c = ISCodeCitation(code="IS 999", version="9999", section="1.1")
    verified = verify_citation(c, registry=registry)
    assert verified.resolved is False
    assert "no chunk found" in verified.rejection_reason
