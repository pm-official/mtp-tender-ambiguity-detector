"""Offline tests for the pipeline orchestrator (Feature B)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from src.pipeline import pipeline_status


@pytest.fixture
def tmp_corpus():
    d = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_pipeline_status_handles_empty_tender(tmp_corpus):
    folder = tmp_corpus / "EMPTY"
    folder.mkdir()
    status = pipeline_status(folder)
    assert status["tender_id"] == "EMPTY"
    # Every artefact should be marked absent
    for label, info in status.items():
        if isinstance(info, dict) and "present" in info:
            assert info["present"] is False


def test_pipeline_status_reports_present_files(tmp_corpus):
    folder = tmp_corpus / "T1"
    parsed = folder / "parsed"
    parsed.mkdir(parents=True)
    (parsed / "clauses.jsonl").write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")

    status = pipeline_status(folder)
    assert status["parsed/clauses.jsonl"]["present"] is True
    assert status["parsed/clauses.jsonl"]["lines"] == 2
    assert status["parsed/clauses.jsonl"]["size_bytes"] > 0   # platform-dependent
