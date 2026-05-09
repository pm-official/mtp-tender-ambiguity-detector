"""Offline tests for the BCT pipeline (no live LLM calls)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.schemas.bct import BCTOutput, Commitment
from src.schemas.clause import Clause


@pytest.fixture
def tmp_corpus():
    d = Path(tempfile.mkdtemp(prefix="bct_pipeline_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_clauses_jsonl(folder: Path, clauses: list[Clause]) -> None:
    parsed = folder / "parsed"
    parsed.mkdir(parents=True, exist_ok=True)
    p = parsed / "clauses.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for c in clauses:
            f.write(c.model_dump_json() + "\n")


def _clause(text: str, *, clause_id="c1", page=1, table=False, n_chars=None):
    return Clause(
        clause_id=clause_id,
        doc_id="doc.pdf",
        tender_id="T1",
        page=page,
        page_end=page,
        section_path=[],
        clause_number=None,
        text=text,
        char_count=n_chars or len(text),
        word_count=len(text.split()),
        table_origin=table,
    )


def test_pipeline_skips_table_origin_by_default(tmp_corpus):
    """Tables are skipped unless include_tables=True."""
    folder = tmp_corpus / "T1"
    folder.mkdir()
    _make_clauses_jsonl(folder, [
        _clause("This is an obligation that the Contractor shall perform.", clause_id="c1"),
        _clause("Sl.No: 1; Description: Item; Unit: cum", clause_id="c2", table=True, n_chars=40),
    ])

    # Mock both the obligation filter and BCT to avoid live calls
    from src.score import pipeline as ppl
    from src.schemas.bct import ObligationCheck

    fake_filter = {"c1": ObligationCheck(clause_id="c1", is_obligation=True, rationale="ok")}
    fake_bct = BCTOutput(commitments=[Commitment(
        obligation="do thing",
        quantity="365 days",
        method="m",
        standard="s",
    )])

    with patch.object(ppl, "filter_obligations", return_value=fake_filter), \
         patch.object(ppl, "run_bct_on_clause") as mock_bct:
        from src.llm.gemini_client import LLMCallStats
        mock_bct.return_value = (fake_bct, LLMCallStats(model="m", cache_hit=True))
        summary = ppl.run_stage1(folder)

    assert summary["skipped_tables"] == 1
    assert summary["eligible"] == 1
    # Only c1 ran through BCT
    assert mock_bct.call_count == 1


def test_pipeline_skips_tiny_clauses(tmp_corpus):
    folder = tmp_corpus / "T2"
    folder.mkdir()
    _make_clauses_jsonl(folder, [
        _clause("ok", clause_id="c1", n_chars=2),  # too tiny
        _clause("This clause is plenty long enough to pass the minimum length filter for sure.", clause_id="c2"),
    ])

    from src.score import pipeline as ppl
    from src.schemas.bct import ObligationCheck

    with patch.object(ppl, "filter_obligations",
                       return_value={"c2": ObligationCheck(clause_id="c2", is_obligation=True, rationale="ok")}), \
         patch.object(ppl, "run_bct_on_clause") as mock_bct:
        from src.llm.gemini_client import LLMCallStats
        mock_bct.return_value = (BCTOutput(commitments=[]), LLMCallStats(model="m", cache_hit=True))
        summary = ppl.run_stage1(folder, min_clause_chars=40)

    assert summary["skipped_tiny"] == 1
    assert summary["eligible"] == 1


def test_pipeline_records_non_obligation_clauses(tmp_corpus):
    """Clauses that the filter says are not obligations should still be in the output (audit trail)."""
    folder = tmp_corpus / "T3"
    folder.mkdir()
    _make_clauses_jsonl(folder, [
        _clause("This is a definition: 'X' means Y for the purposes of this contract.", clause_id="c1"),
    ])

    from src.score import pipeline as ppl
    from src.schemas.bct import ObligationCheck

    with patch.object(
        ppl, "filter_obligations",
        return_value={"c1": ObligationCheck(clause_id="c1", is_obligation=False, rationale="definition")},
    ), patch.object(ppl, "run_bct_on_clause") as mock_bct:
        summary = ppl.run_stage1(folder)
        mock_bct.assert_not_called()  # BCT shouldn't run on non-obligations

    assert summary["bct_calls"] == 0
    # The clause should still be recorded as a non-obligation
    flags = ppl.load_flags(folder)
    assert len(flags) == 1
    assert flags[0].is_obligation is False
    assert flags[0].flagged is False


def test_load_flags_returns_empty_when_missing(tmp_corpus):
    from src.score.pipeline import load_flags
    folder = tmp_corpus / "missing"
    assert load_flags(folder) == []


def test_pipeline_writes_summary_json(tmp_corpus):
    folder = tmp_corpus / "T4"
    folder.mkdir()
    _make_clauses_jsonl(folder, [
        _clause("The Contractor shall complete the Works within 365 days.", clause_id="c1"),
    ])

    from src.score import pipeline as ppl
    from src.schemas.bct import ObligationCheck
    from src.llm.gemini_client import LLMCallStats

    with patch.object(ppl, "filter_obligations",
                       return_value={"c1": ObligationCheck(clause_id="c1", is_obligation=True)}), \
         patch.object(ppl, "run_bct_on_clause") as mock_bct:
        mock_bct.return_value = (
            BCTOutput(commitments=[Commitment(
                obligation="complete the Works",
                quantity="365 days",
                method="completion certificate",
                standard="CONTRACT_DEFINED",
            )]),
            LLMCallStats(model="m", cache_hit=True),
        )
        ppl.run_stage1(folder)

    summary_path = folder / "parsed" / "stage1_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["bct_calls"] == 1
    assert summary["flagged"] == 0
    assert summary["not_flagged"] == 1
