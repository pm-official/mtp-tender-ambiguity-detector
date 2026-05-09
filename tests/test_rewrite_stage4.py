"""Offline tests for Stage 4 rewrite (mocked LLM + embed)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.llm.gemini_client import LLMCallStats
from src.schemas.bct import BCTOutput, CandidateFlag, Commitment
from src.schemas.rewrite import ISCodeChunk
from src.schemas.stage3 import ConfirmedFlag, SeverityScore


def _flag_and_confirmed():
    flag = CandidateFlag(
        flag_id="d.pdf::flag::00001",
        clause_id="d.pdf::00001",
        doc_id="d.pdf",
        tender_id="T1",
        page=5,
        clause_text="The Contractor shall maintain reasonable dust suppression during demolition.",
        section_path=["SCC"],
        clause_number="5.3",
        bct_output=BCTOutput(commitments=[Commitment(
            obligation="maintain dust suppression",
            quantity="CANNOT_DETERMINE",
            method="CANNOT_DETERMINE",
            standard="CANNOT_DETERMINE",
            missing_info=["What PM10 limit applies?"],
        )]),
        flagged=True,
        flag_reason="vague",
    )
    confirmed = ConfirmedFlag(
        flag_id=flag.flag_id,
        clause_id=flag.clause_id,
        doc_id=flag.doc_id,
        tender_id=flag.tender_id,
        page=flag.page,
        clause_text=flag.clause_text,
        stage2_verdict="REAL",
        critique_verdict="CONFIRMED",
        severity=SeverityScore(commercial_exposure=4, dispute_likelihood=4, reviewer_cost=3),
        composite_severity=8,
        severity_tier="high",
    )
    return flag, confirmed


def test_rewrite_passes_all_three_guardrails():
    """Happy path: rewrite is good, all guardrails pass, status=ACCEPTED."""
    from src.rewrite import stage4
    from src.schemas.rewrite import ISCodeCitation

    flag, confirmed = _flag_and_confirmed()

    fake_iscode = [ISCodeChunk(
        chunk_id="x", code_id="IS 5182:1999", code_title="Air Quality",
        version_year=1999, section="5.2", text="PM10 ≤ 100 µg/m³ over 24h", char_count=20,
    )]
    fake_proposal = stage4._RewriteProposal(
        rewrite_text=("The Contractor shall maintain dust suppression such that PM10 "
                      "concentrations measured at the site boundary do not exceed 100 µg/m³ "
                      "over a 24-hour period, in accordance with IS 5182:1999 Section 5.2."),
        is_code_citations=[stage4._ProposedCitation(code="IS 5182", version="1999", section="5.2")],
        explanation="Replaced 'reasonable' with PM10 limit + IS 5182 reference.",
        preserves_intent=True,
    )
    fake_stats = LLMCallStats(model="x", cache_hit=True)

    # Mock retrieval, embed, LLM, BCT, citation verification.
    with patch.object(stage4, "query_iscode_index", return_value=fake_iscode), \
         patch.object(stage4, "verify_all_citations") as mock_verify, \
         patch.object(stage4, "run_bct_on_clause") as mock_bct, \
         patch.object(stage4, "get_default_client") as mock_client_fn:

        # Verify mock returns one resolved citation
        mock_verify.return_value = (
            [ISCodeCitation(code="IS 5182", version="1999", section="5.2",
                             raw_text="IS 5182:1999 §5.2", resolved=True,
                             resolved_chunk_id="x")],
            1, 0,
        )
        # BCT re-check passes (no CANNOT_DETERMINE → not flagged)
        from src.schemas.bct import BCTOutput as BO
        mock_bct.return_value = (
            BO(commitments=[Commitment(obligation="maintain", quantity="100 µg/m³",
                                         method="boundary measurement", standard="IS 5182:1999")]),
            fake_stats,
        )
        # LLM client returns the proposal; embedding returns matching vector for high cosine
        client = mock_client_fn.return_value
        client.default_pro = "gemini-2.5-pro"
        client.default_embedding = "gemini-embedding-001"
        # 1) embed of original_text
        # 2) generate rewrite proposal
        # 3) embed of rewrite text (close vector → cosine high)
        client.embed.side_effect = [
            ([[1.0, 0.0, 0.0]], fake_stats),         # original embed
            ([[0.99, 0.01, 0.0]], fake_stats),        # rewrite embed
        ]
        client.generate.return_value = (fake_proposal, fake_stats)

        rewrite, _ = stage4.rewrite_one(confirmed, flag)

    assert rewrite.status == "ACCEPTED"
    assert rewrite.attempts == 1
    assert rewrite.citation_integrity_passed is True
    assert rewrite.reflag_check_passed is True
    assert rewrite.intent_check_passed is True
    assert rewrite.intent_cosine > 0.95


def test_rewrite_fails_g1_then_succeeds_on_retry():
    """Citation hallucination on attempt 1, fixed on attempt 2."""
    from src.rewrite import stage4
    from src.schemas.rewrite import ISCodeCitation

    flag, confirmed = _flag_and_confirmed()

    fake_iscode = [ISCodeChunk(
        chunk_id="x", code_id="IS 5182:1999", code_title="Air Quality",
        version_year=1999, section="5.2", text="...", char_count=10,
    )]
    bad_proposal = stage4._RewriteProposal(
        rewrite_text="The Contractor shall do X per IS 9999 Section 99.",
        is_code_citations=[stage4._ProposedCitation(code="IS 9999", version=None, section="99")],
        explanation="bad",
        preserves_intent=True,
    )
    good_proposal = stage4._RewriteProposal(
        rewrite_text="The Contractor shall maintain PM10 ≤ 100 µg/m³ per IS 5182:1999 §5.2.",
        is_code_citations=[stage4._ProposedCitation(code="IS 5182", version="1999", section="5.2")],
        explanation="fixed",
        preserves_intent=True,
    )
    fake_stats = LLMCallStats(model="x", cache_hit=True)

    with patch.object(stage4, "query_iscode_index", return_value=fake_iscode), \
         patch.object(stage4, "verify_all_citations") as mock_verify, \
         patch.object(stage4, "run_bct_on_clause") as mock_bct, \
         patch.object(stage4, "get_default_client") as mock_client_fn:

        # First verify call: 0 resolved (bad). Second: 1 resolved (good).
        mock_verify.side_effect = [
            ([ISCodeCitation(code="IS 9999", section="99", raw_text="IS 9999 §99", resolved=False,
                              rejection_reason="not in registry")], 0, 1),
            ([ISCodeCitation(code="IS 5182", version="1999", section="5.2", raw_text="IS 5182:1999 §5.2",
                              resolved=True, resolved_chunk_id="x")], 1, 0),
        ]
        from src.schemas.bct import BCTOutput as BO
        mock_bct.return_value = (
            BO(commitments=[Commitment(obligation="maintain", quantity="100 µg/m³",
                                         method="boundary", standard="IS 5182:1999")]),
            fake_stats,
        )
        client = mock_client_fn.return_value
        client.default_pro = "gemini-2.5-pro"
        client.default_embedding = "gemini-embedding-001"
        client.embed.side_effect = [
            ([[1.0, 0.0]], fake_stats),
            ([[0.95, 0.05]], fake_stats),
        ]
        client.generate.side_effect = [
            (bad_proposal, fake_stats),
            (good_proposal, fake_stats),
        ]

        rewrite, _ = stage4.rewrite_one(confirmed, flag, max_attempts=3)

    assert rewrite.status == "ACCEPTED"
    assert rewrite.attempts == 2  # succeeded on 2nd attempt


def test_rewrite_fails_g2_until_retry_limit():
    """Rewrite REGRESSES on BCT (more CANNOT_DETERMINE than original) — stop
    after max_attempts. Under differential G2, only a regression fails."""
    from src.rewrite import stage4
    from src.schemas.rewrite import ISCodeCitation

    flag, confirmed = _flag_and_confirmed()
    # Original has 1 commitment × 3 CDs = 3 CDs. To fail differential G2 the
    # rewrite must produce STRICTLY MORE CDs.

    fake_iscode = [ISCodeChunk(chunk_id="x", code_id="IS 456:2000", code_title="RCC",
                                 version_year=2000, section="5", text="...", char_count=5)]
    proposal = stage4._RewriteProposal(
        rewrite_text="The Contractor shall maintain dust suppression per IS 456:2000 §5.",
        is_code_citations=[stage4._ProposedCitation(code="IS 456", version="2000", section="5")],
        explanation="...",
        preserves_intent=True,
    )
    fake_stats = LLMCallStats(model="x", cache_hit=True)

    with patch.object(stage4, "query_iscode_index", return_value=fake_iscode), \
         patch.object(stage4, "verify_all_citations") as mock_verify, \
         patch.object(stage4, "run_bct_on_clause") as mock_bct, \
         patch.object(stage4, "get_default_client") as mock_client_fn:

        mock_verify.return_value = (
            [ISCodeCitation(code="IS 456", version="2000", section="5",
                             raw_text="IS 456:2000 §5", resolved=True, resolved_chunk_id="x")],
            1, 0,
        )
        # Regression: rewrite extracts 2 commitments, each with 3 CDs = 6 CDs
        # vs original's 3. Differential G2 should fail.
        from src.schemas.bct import BCTOutput as BO
        bct_regress_output = BO(commitments=[
            Commitment(obligation="maintain", quantity="CANNOT_DETERMINE",
                       method="CANNOT_DETERMINE", standard="CANNOT_DETERMINE"),
            Commitment(obligation="report", quantity="CANNOT_DETERMINE",
                       method="CANNOT_DETERMINE", standard="CANNOT_DETERMINE"),
        ])
        mock_bct.return_value = (bct_regress_output, fake_stats)
        client = mock_client_fn.return_value
        client.default_pro = "gemini-2.5-pro"
        client.default_embedding = "gemini-embedding-001"
        client.embed.side_effect = [([[1.0, 0.0]], fake_stats)] + [([[0.99, 0.01]], fake_stats)] * 6
        client.generate.return_value = (proposal, fake_stats)

        rewrite, _ = stage4.rewrite_one(confirmed, flag, max_attempts=3)

    # All 3 attempts fail G2 (regression), end status reflects that
    assert rewrite.attempts == 3
    assert rewrite.reflag_check_passed is False
    assert rewrite.original_cd_count == 3
    assert rewrite.rewrite_cd_count == 6
    assert rewrite.status in ("RETRY_LIMIT_REACHED", "PASSED_GUARDRAILS_FAILED")


def test_rewrite_g2_passes_when_cd_count_does_not_regress():
    """Differential G2: rewrite with same or fewer CDs than original passes,
    even if some fields are still CANNOT_DETERMINE. This catches the
    over-decomposition artifact where Stage 1 BCT extracts a continuation
    property as a separate sub-obligation."""
    from src.rewrite import stage4
    from src.schemas.rewrite import ISCodeCitation

    flag, confirmed = _flag_and_confirmed()
    # Original: 1 commitment × 3 CDs = 3 CDs

    fake_iscode = [ISCodeChunk(chunk_id="x", code_id="IS 456:2000", code_title="RCC",
                                 version_year=2000, section="5", text="...", char_count=5)]
    proposal = stage4._RewriteProposal(
        rewrite_text=("The Contractor shall maintain dust suppression at PM10 ≤ "
                      "100 µg/m³ measured at the boundary, per IS 5182:1999 §5.2. "
                      "Measurements shall be reported monthly."),
        is_code_citations=[stage4._ProposedCitation(code="IS 5182", version="1999", section="5.2")],
        explanation="filled gaps; added reporting cadence",
        preserves_intent=True,
    )
    fake_stats = LLMCallStats(model="x", cache_hit=True)

    with patch.object(stage4, "query_iscode_index", return_value=fake_iscode), \
         patch.object(stage4, "verify_all_citations") as mock_verify, \
         patch.object(stage4, "run_bct_on_clause") as mock_bct, \
         patch.object(stage4, "get_default_client") as mock_client_fn:

        mock_verify.return_value = (
            [ISCodeCitation(code="IS 5182", version="1999", section="5.2",
                             raw_text="IS 5182:1999 §5.2", resolved=True, resolved_chunk_id="x")],
            1, 0,
        )
        # Stage 1 over-decomposes the rewrite: primary obligation is fully
        # specified (0 CDs) but a "report monthly" sub-obligation has 2 CDs
        # (method, standard). Total = 2 CDs vs original's 3 → non-regression.
        from src.schemas.bct import BCTOutput as BO
        bct_overdecomp = BO(commitments=[
            Commitment(obligation="maintain dust suppression",
                       quantity="100 µg/m³", method="boundary measurement",
                       standard="IS 5182:1999"),
            Commitment(obligation="report monthly",
                       quantity="monthly", method="CANNOT_DETERMINE",
                       standard="CANNOT_DETERMINE"),
        ])
        mock_bct.return_value = (bct_overdecomp, fake_stats)
        client = mock_client_fn.return_value
        client.default_pro = "gemini-2.5-pro"
        client.default_embedding = "gemini-embedding-001"
        client.embed.side_effect = [
            ([[1.0, 0.0, 0.0]], fake_stats),
            ([[0.99, 0.01, 0.0]], fake_stats),
        ]
        client.generate.return_value = (proposal, fake_stats)

        rewrite, _ = stage4.rewrite_one(confirmed, flag)

    assert rewrite.status == "ACCEPTED"
    assert rewrite.original_cd_count == 3
    assert rewrite.rewrite_cd_count == 2
    assert rewrite.reflag_check_passed is True


def test_rewrite_skips_g3_gracefully_when_embed_fails():
    """If embedding API is unavailable, G3 is skipped (intent_check_passed=None)
    and the rewrite is judged on G1 + G2 alone."""
    from src.rewrite import stage4
    from src.schemas.rewrite import ISCodeCitation

    flag, confirmed = _flag_and_confirmed()

    proposal = stage4._RewriteProposal(
        rewrite_text="The Contractor shall maintain PM10 ≤ 100 µg/m³ per IS 5182:1999 §5.2.",
        is_code_citations=[stage4._ProposedCitation(code="IS 5182", version="1999", section="5.2")],
        explanation="...",
        preserves_intent=True,
    )
    fake_stats = LLMCallStats(model="x", cache_hit=True)

    with patch.object(stage4, "query_iscode_index", return_value=[]), \
         patch.object(stage4, "verify_all_citations") as mock_verify, \
         patch.object(stage4, "run_bct_on_clause") as mock_bct, \
         patch.object(stage4, "get_default_client") as mock_client_fn:

        mock_verify.return_value = (
            [ISCodeCitation(code="IS 5182", version="1999", section="5.2",
                             raw_text="IS 5182:1999 §5.2", resolved=True, resolved_chunk_id="x")],
            1, 0,
        )
        from src.schemas.bct import BCTOutput as BO
        mock_bct.return_value = (
            BO(commitments=[Commitment(obligation="maintain", quantity="100 µg/m³",
                                         method="boundary", standard="IS 5182:1999")]),
            fake_stats,
        )
        client = mock_client_fn.return_value
        client.default_pro = "gemini-2.5-pro"
        client.default_embedding = "gemini-embedding-001"
        # Both embed calls fail with 403
        client.embed.side_effect = Exception("403 PERMISSION_DENIED")
        client.generate.return_value = (proposal, fake_stats)

        rewrite, _ = stage4.rewrite_one(confirmed, flag)

    # G3 skipped → None; rewrite still ACCEPTED on G1 + G2
    assert rewrite.intent_check_passed is None
    assert rewrite.citation_integrity_passed is True
    assert rewrite.reflag_check_passed is True
    assert rewrite.status == "ACCEPTED"
