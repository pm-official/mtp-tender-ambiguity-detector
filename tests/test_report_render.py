"""Offline tests for the HTML report renderer (Feature C)."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.report.render import render_html_report
from src.schemas.bct import BCTOutput, CandidateFlag, Commitment
from src.schemas.stage2 import Stage2Verdict


@pytest.fixture
def tmp_corpus():
    d = Path(tempfile.mkdtemp(prefix="report_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _setup_minimal_tender(folder: Path) -> None:
    parsed = folder / "parsed"
    parsed.mkdir(parents=True)

    # One flagged + one not-flagged
    flag1 = CandidateFlag(
        flag_id="d.pdf::flag::00001",
        clause_id="d.pdf::00001",
        doc_id="d.pdf",
        tender_id=folder.name,
        page=1,
        clause_text="The Contractor shall maintain reasonable dust suppression.",
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
    flag2 = CandidateFlag(
        flag_id="d.pdf::flag::00002",
        clause_id="d.pdf::00002",
        doc_id="d.pdf",
        tender_id=folder.name,
        page=2,
        clause_text="The Contractor shall complete the Works within 365 days.",
        section_path=["GCC"],
        clause_number="4.2",
        bct_output=BCTOutput(commitments=[Commitment(
            obligation="complete the Works",
            quantity="365 days",
            method="completion certificate",
            standard="CONTRACT_DEFINED",
        )]),
        flagged=False,
        flag_reason="all concrete",
    )
    with (parsed / "candidate_flags.jsonl").open("w", encoding="utf-8") as f:
        f.write(flag1.model_dump_json() + "\n")
        f.write(flag2.model_dump_json() + "\n")

    verdict = Stage2Verdict(
        flag_id=flag1.flag_id,
        clause_id=flag1.clause_id,
        verdict="REAL",
        rationale="No retrieved chunk supplies a PM10 limit.",
        confidence=0.9,
    )
    with (parsed / "stage2_verdicts.jsonl").open("w", encoding="utf-8") as f:
        f.write(verdict.model_dump_json() + "\n")


def test_render_writes_html(tmp_corpus):
    folder = tmp_corpus / "T1"
    folder.mkdir()
    _setup_minimal_tender(folder)
    out = render_html_report(folder)
    assert out.exists()
    assert out.suffix == ".html"
    text = out.read_text(encoding="utf-8")
    assert "<html" in text
    assert "Tender Ambiguity Report" in text


def test_render_includes_clause_text_and_bct_output(tmp_corpus):
    folder = tmp_corpus / "T1"
    folder.mkdir()
    _setup_minimal_tender(folder)
    out = render_html_report(folder)
    text = out.read_text(encoding="utf-8")
    assert "reasonable dust suppression" in text
    assert "What PM10 limit applies?" in text
    assert "CANNOT_DETERMINE" in text


def test_render_shows_verdict_pill(tmp_corpus):
    folder = tmp_corpus / "T1"
    folder.mkdir()
    _setup_minimal_tender(folder)
    out = render_html_report(folder)
    text = out.read_text(encoding="utf-8")
    assert "verdict-pill real" in text
    assert "REAL" in text
    assert "No retrieved chunk supplies a PM10 limit." in text


def test_render_aggregates_pre_bid_questions(tmp_corpus):
    folder = tmp_corpus / "T1"
    folder.mkdir()
    _setup_minimal_tender(folder)
    out = render_html_report(folder)
    text = out.read_text(encoding="utf-8")
    assert "Pre-bid clarification letter" in text


def test_render_works_with_no_flags(tmp_corpus):
    """Empty tender should render without crashing."""
    folder = tmp_corpus / "EMPTY"
    parsed = folder / "parsed"
    parsed.mkdir(parents=True)
    (parsed / "candidate_flags.jsonl").write_text("", encoding="utf-8")
    (parsed / "stage2_verdicts.jsonl").write_text("", encoding="utf-8")

    out = render_html_report(folder)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Tender Ambiguity Report" in text
