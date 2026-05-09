"""Offline tests for the Stage 2 critic.

The critic itself is an LLM call; we mock it to test the orchestration logic.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.critic.stage2 import (
    _format_bct_findings,
    _format_retrieved_block,
    run_stage2_critic,
)
from src.llm.gemini_client import LLMCallStats
from src.schemas.bct import BCTOutput, CandidateFlag, Commitment
from src.schemas.stage2 import HybridHit, Stage2Verdict


_DEFAULT_COMMITMENT = Commitment(
    obligation="maintain dust suppression",
    quantity="CANNOT_DETERMINE",
    method="CANNOT_DETERMINE",
    standard="CANNOT_DETERMINE",
    missing_info=["What PM10 limit?"],
)


def _make_flag(*, flag_id="f1", clause_id="c1", commitments=...) -> CandidateFlag:
    """Sentinel-based default lets callers pass [] explicitly."""
    if commitments is ...:
        commitments = [_DEFAULT_COMMITMENT]
    return CandidateFlag(
        flag_id=flag_id,
        clause_id=clause_id,
        doc_id="d.pdf",
        tender_id="T1",
        page=5,
        clause_text="The Contractor shall maintain reasonable dust suppression.",
        section_path=["Section 5"],
        clause_number="5.3",
        bct_output=BCTOutput(commitments=commitments),
        flagged=True,
        flag_reason="vagueness",
    )


def _make_hit(*, clause_id="cx", text="some retrieved text", routes=None, rank=1) -> HybridHit:
    return HybridHit(
        clause_id=clause_id,
        text=text,
        metadata={"page": 7, "section": "Specs", "clause_number": "6.1"},
        rrf_score=0.05,
        rank=rank,
        sources=routes or ["R1_dense"],
    )


# ─── Prompt building helpers ────────────────────────────────────────────────
def test_format_bct_findings_lists_each_commitment():
    flag = _make_flag(commitments=[
        Commitment(obligation="o1", quantity="365 days", method="CANNOT_DETERMINE", standard="IS 456:2000"),
        Commitment(obligation="o2", quantity="CANNOT_DETERMINE", method="m", standard="s",
                   missing_info=["Q1?", "Q2?"]),
    ])
    s = _format_bct_findings(flag)
    assert "Commitment 1" in s
    assert "Commitment 2" in s
    assert "method" in s   # commitment 1 has method=CANNOT_DETERMINE
    assert "Q1?" in s


def test_format_bct_findings_handles_empty():
    flag = _make_flag(commitments=[])
    s = _format_bct_findings(flag)
    assert "no commitments" in s.lower()


def test_format_retrieved_block_lists_hits():
    hits = [_make_hit(clause_id="hA", rank=1), _make_hit(clause_id="hB", rank=2)]
    s = _format_retrieved_block(hits)
    assert "hA" in s
    assert "hB" in s


def test_format_retrieved_block_truncates_long_text():
    long_text = "x" * 2000
    hits = [_make_hit(text=long_text)]
    s = _format_retrieved_block(hits, max_chars=600)
    assert "..." in s


# ─── Critic call ────────────────────────────────────────────────────────────
def test_critic_patches_flag_and_clause_ids():
    """The critic should inject flag_id and clause_id into the returned verdict
    even if the LLM didn't include them in its JSON output."""
    flag = _make_flag(flag_id="myFlag", clause_id="myClause")
    hits = [_make_hit(clause_id="hA")]

    fake_verdict = Stage2Verdict(
        flag_id="",   # LLM may have left this empty
        clause_id="",
        verdict="REAL",
        rationale="The retrieved chunks do not resolve the missing PM10 limit.",
        confidence=0.85,
    )
    fake_stats = LLMCallStats(
        model="gemini-2.5-flash-lite",
        cache_hit=True,
        cost_inr=0.01,
    )

    with patch("src.critic.stage2.get_default_client") as mock_client_fn:
        client = mock_client_fn.return_value
        client.default_flash = "gemini-2.5-flash-lite"
        client.generate.return_value = (fake_verdict, fake_stats)
        verdict, stats = run_stage2_critic(flag, hits)

    assert verdict.flag_id == "myFlag"
    assert verdict.clause_id == "myClause"
    assert verdict.verdict == "REAL"
    assert verdict.confidence == 0.85
    assert verdict.retrieved_chunk_ids == ["hA"]
    assert verdict.retrieved_sources == [["R1_dense"]]


def test_critic_coerces_unknown_verdict_to_real():
    flag = _make_flag()
    hits = [_make_hit()]

    fake_verdict = Stage2Verdict(
        flag_id="",
        clause_id="",
        verdict="UNCERTAIN",   # not a valid literal — pydantic itself would catch most
        rationale="...",
        confidence=0.5,
    ) if False else None  # skip if pydantic rejects it; use REAL+coercion path instead

    # We cannot construct an invalid literal under strict pydantic; instead
    # simulate a valid-but-unexpected literal by patching evaluate
    fake_verdict = Stage2Verdict(
        flag_id="", clause_id="", verdict="REAL",
        rationale="r", confidence=0.5,
    )
    # Mutate after construction to simulate an LLM returning "FOO"
    object.__setattr__(fake_verdict, "verdict", "FOO")

    fake_stats = LLMCallStats(model="x", cache_hit=True, cost_inr=0.0)

    with patch("src.critic.stage2.get_default_client") as mock_client_fn:
        client = mock_client_fn.return_value
        client.default_flash = "gemini-2.5-flash-lite"
        client.generate.return_value = (fake_verdict, fake_stats)
        verdict, stats = run_stage2_critic(flag, hits)

    assert verdict.verdict == "REAL"
