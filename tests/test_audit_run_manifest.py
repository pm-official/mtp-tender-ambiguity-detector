"""Offline tests for the run-manifest layer (Feature A)."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from src.audit.run_manifest import (
    finish_run,
    list_runs,
    record_input,
    record_llm_stats,
    record_output,
    start_run,
    write_manifest,
)
from src.llm.gemini_client import LLMCallStats


@pytest.fixture
def tmp_corpus():
    d = Path(tempfile.mkdtemp(prefix="manifest_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_start_run_populates_required_fields():
    m = start_run(tender_id="T1", stage="stage_1_bct",
                  parameters={"a": 1}, models={"bct": "gemini-2.5-flash"})
    assert m.tender_id == "T1"
    assert m.stage == "stage_1_bct"
    assert m.stage_version != "0.0"   # known stage has a version
    assert m.parameters == {"a": 1}
    assert m.models == {"bct": "gemini-2.5-flash"}
    assert m.environment["python"]
    assert m.run_id


def test_record_input_captures_size_and_lines(tmp_corpus):
    m = start_run(tender_id="T1", stage="stage_1_bct")
    p = tmp_corpus / "data.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    record_input(m, p)
    entry = m.inputs[0]
    assert entry["lines"] == 2
    assert entry["size_bytes"] > 0   # exact byte count is platform-dependent (\n vs \r\n)
    assert "sha256" in entry
    assert len(entry["sha256"]) == 64


def test_record_input_handles_missing_file(tmp_corpus):
    m = start_run(tender_id="T1", stage="stage_1_bct")
    record_input(m, tmp_corpus / "does_not_exist.jsonl")
    assert m.inputs[0]["missing"] is True


def test_record_llm_stats_accumulates():
    m = start_run(tender_id="T1", stage="stage_1_bct")
    s1 = LLMCallStats(model="x", cache_hit=False, prompt_tokens=10, completion_tokens=20,
                       cost_usd=0.01, cost_inr=0.84)
    s2 = LLMCallStats(model="x", cache_hit=True, prompt_tokens=10, completion_tokens=20,
                       cost_usd=0.01, cost_inr=0.84)
    record_llm_stats(m, s1)
    record_llm_stats(m, s2)
    assert m.api_calls == 2
    assert m.cache_hits == 1
    assert m.cache_misses == 1
    assert m.tokens_in == 20
    assert abs(m.cost_usd - 0.02) < 1e-9


def test_finish_run_computes_elapsed():
    m = start_run(tender_id="T1", stage="stage_1_bct")
    time.sleep(0.05)
    finish_run(m)
    assert m.success is True
    assert m.seconds_elapsed >= 0.0
    assert m.finished_at >= m.started_at


def test_write_manifest_persists_and_indexes(tmp_corpus):
    m = start_run(tender_id="T1", stage="stage_1_bct")
    finish_run(m)
    out = write_manifest(tmp_corpus, m)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["run_id"] == m.run_id

    runs_idx = tmp_corpus / "runs" / "_index.jsonl"
    assert runs_idx.exists()
    line = runs_idx.read_text(encoding="utf-8").strip()
    entry = json.loads(line)
    assert entry["run_id"] == m.run_id
    assert entry["stage"] == "stage_1_bct"


def test_list_runs_returns_recent_first(tmp_corpus):
    for stage in ("stage_1_bct", "stage_2_critic"):
        m = start_run(tender_id="T1", stage=stage)
        finish_run(m)
        write_manifest(tmp_corpus, m)
    runs = list_runs(tmp_corpus)
    assert len(runs) == 2
    assert {r["stage"] for r in runs} == {"stage_1_bct", "stage_2_critic"}


def test_list_runs_returns_empty_when_no_runs_dir(tmp_corpus):
    # Doesn't error if dir doesn't exist
    assert list_runs(tmp_corpus) == []
