"""Offline tests for the Stage 3 pipeline (mocked LLM calls)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.llm.gemini_client import LLMCallStats
from src.schemas.bct import BCTOutput, CandidateFlag, Commitment
from src.schemas.clause import Clause
from src.schemas.stage2 import Stage2Verdict
from src.schemas.stage3 import CritiqueVerdict, SeverityScore


@pytest.fixture
def tmp_corpus():
    d = Path(tempfile.mkdtemp(prefix="stage3_pipeline_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _setup_fixtures(folder: Path) -> None:
    parsed = folder / "parsed"
    parsed.mkdir(parents=True)

    # 1 clause
    c = Clause(
        clause_id="d.pdf::00001",
        doc_id="d.pdf",
        tender_id=folder.name,
        page=1,
        page_end=1,
        section_path=["SCC"],
        clause_number="5.3",
        text="The Contractor shall maintain reasonable dust suppression.",
        char_count=58,
        word_count=8,
    )
    with (parsed / "clauses.jsonl").open("w", encoding="utf-8") as f:
        f.write(c.model_dump_json() + "\n")

    # 1 flagged + 1 not-flagged candidate
    flag1 = CandidateFlag(
        flag_id="d.pdf::flag::00001",
        clause_id="d.pdf::00001",
        doc_id="d.pdf",
        tender_id=folder.name,
        page=1,
        clause_text=c.text,
        section_path=c.section_path,
        clause_number=c.clause_number,
        bct_output=BCTOutput(commitments=[Commitment(
            obligation="maintain dust suppression",
            quantity="CANNOT_DETERMINE",
            method="CANNOT_DETERMINE",
            standard="CANNOT_DETERMINE",
            missing_info=["What PM10 limit?"],
        )]),
        flagged=True,
        flag_reason="vague",
    )
    flag2 = CandidateFlag(
        flag_id="d.pdf::flag::00002",
        clause_id="d.pdf::00001",
        doc_id="d.pdf",
        tender_id=folder.name,
        page=1,
        clause_text="(non-obligation)",
        bct_output=BCTOutput(commitments=[]),
        flagged=False,
        flag_reason="not an obligation",
        is_obligation=False,
    )
    with (parsed / "candidate_flags.jsonl").open("w", encoding="utf-8") as f:
        f.write(flag1.model_dump_json() + "\n")
        f.write(flag2.model_dump_json() + "\n")

    # Stage-2 verdicts: one REAL (the flagged), one APPARENT
    v1 = Stage2Verdict(
        flag_id="d.pdf::flag::00001",
        clause_id="d.pdf::00001",
        verdict="REAL",
        rationale="No retrieved chunk supplies a PM10 limit.",
        confidence=0.9,
    )
    v2 = Stage2Verdict(
        flag_id="d.pdf::flag::00002",
        clause_id="d.pdf::00001",
        verdict="APPARENT",
        rationale="Resolved by another chunk.",
        confidence=0.85,
    )
    with (parsed / "stage2_verdicts.jsonl").open("w", encoding="utf-8") as f:
        f.write(v1.model_dump_json() + "\n")
        f.write(v2.model_dump_json() + "\n")


def test_stage3_critiques_only_real_verdicts(tmp_corpus):
    """Stage 3 must only critique REAL verdicts. APPARENT/WEAK pass through."""
    folder = tmp_corpus / "T1"
    _setup_fixtures(folder)

    from src.critic import stage3_pipeline as ppl

    fake_critique = CritiqueVerdict(
        flag_id="", clause_id="",
        verdict="CONFIRMED",
        refutation="Genuine vagueness; cannot commit.",
        confidence=0.85,
    )
    fake_severity = SeverityScore(
        commercial_exposure=4, dispute_likelihood=4, reviewer_cost=3,
        rationale="Material to BoQ.",
    )
    fake_stats = LLMCallStats(model="x", cache_hit=True, cost_inr=0.01)

    with patch.object(ppl, "retrieve_for_flag", return_value=[]), \
         patch.object(ppl, "run_critique_pass", return_value=(fake_critique, fake_stats)) as mock_crit, \
         patch.object(ppl, "run_severity_scoring", return_value=(fake_severity, fake_stats)) as mock_sev:
        summary = ppl.run_stage3(folder)

    # Only one REAL verdict → exactly one critique call, exactly one severity call
    assert mock_crit.call_count == 1
    assert mock_sev.call_count == 1
    assert summary["critique_calls"] == 1
    assert summary["critique_confirmed"] == 1
    assert summary["severity_calls"] == 1
    assert summary["total_records"] == 2  # both verdicts pass through


def test_stage3_passes_through_apparent_verdicts(tmp_corpus):
    """APPARENT verdicts should appear in confirmed.jsonl with critique_verdict=None."""
    folder = tmp_corpus / "T2"
    _setup_fixtures(folder)

    from src.critic import stage3_pipeline as ppl

    with patch.object(ppl, "retrieve_for_flag", return_value=[]), \
         patch.object(ppl, "run_critique_pass") as mock_crit:
        mock_crit.return_value = (
            CritiqueVerdict(flag_id="", clause_id="", verdict="CONFIRMED", refutation="ok"),
            LLMCallStats(model="x", cache_hit=True),
        )
        ppl.run_stage3(folder, skip_severity=True)

    confirmed = ppl.load_confirmed(folder)
    apparent = [c for c in confirmed if c.stage2_verdict == "APPARENT"]
    assert len(apparent) == 1
    assert apparent[0].critique_verdict is None


def test_stage3_skip_severity_skips_severity_calls(tmp_corpus):
    folder = tmp_corpus / "T3"
    _setup_fixtures(folder)

    from src.critic import stage3_pipeline as ppl

    fake_critique = CritiqueVerdict(flag_id="", clause_id="", verdict="CONFIRMED",
                                     refutation="ok", confidence=0.8)
    fake_stats = LLMCallStats(model="x", cache_hit=True)

    with patch.object(ppl, "retrieve_for_flag", return_value=[]), \
         patch.object(ppl, "run_critique_pass", return_value=(fake_critique, fake_stats)), \
         patch.object(ppl, "run_severity_scoring") as mock_sev:
        ppl.run_stage3(folder, skip_severity=True)
        mock_sev.assert_not_called()


def test_stage3_writes_summary(tmp_corpus):
    folder = tmp_corpus / "T4"
    _setup_fixtures(folder)

    from src.critic import stage3_pipeline as ppl

    fake_critique = CritiqueVerdict(flag_id="", clause_id="", verdict="CONFIRMED",
                                     refutation="ok", confidence=0.8)
    fake_severity = SeverityScore(commercial_exposure=3, dispute_likelihood=3, reviewer_cost=3)
    fake_stats = LLMCallStats(model="x", cache_hit=True)

    with patch.object(ppl, "retrieve_for_flag", return_value=[]), \
         patch.object(ppl, "run_critique_pass", return_value=(fake_critique, fake_stats)), \
         patch.object(ppl, "run_severity_scoring", return_value=(fake_severity, fake_stats)):
        ppl.run_stage3(folder)

    summary_path = folder / "parsed" / "stage3_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["critique_confirmed"] == 1
    assert summary["severity_distribution"]["medium"] == 1   # 3+3=6 → medium
