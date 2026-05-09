"""End-to-end test: parse the synthetic tender, verify outputs."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.acquire.synthetic import generate_synthetic_tender
from src.parse.pipeline import parse_tender


@pytest.fixture
def parsed_synthetic_dir():
    d = Path(tempfile.mkdtemp(prefix="parse_pipeline_test_"))
    folder = generate_synthetic_tender(d, tender_id="PARSE_TEST_001")
    parse_tender(folder, enable_ocr=False, enable_tables=True)
    yield folder
    shutil.rmtree(d, ignore_errors=True)


def test_parse_writes_clauses_jsonl(parsed_synthetic_dir):
    p = parsed_synthetic_dir / "parsed" / "clauses.jsonl"
    assert p.exists()
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert len(rows) > 0


def test_parse_writes_document_tree(parsed_synthetic_dir):
    p = parsed_synthetic_dir / "parsed" / "document_tree.json"
    assert p.exists()
    tree = json.loads(p.read_text(encoding="utf-8"))
    assert tree["tender_id"] == "PARSE_TEST_001"
    assert tree["summary"]["documents_parsed"] == 7  # 5 base PDFs + 2 corrigenda
    assert tree["summary"]["total_clauses"] >= 5


def test_parse_finds_synthetic_vague_clauses(parsed_synthetic_dir):
    """The synthetic tender has known-vague clauses; verify they survive parsing.

    PyMuPDF preserves newlines inside phrases (e.g. "reasonable\\nlevels"), so we
    normalise whitespace before searching.
    """
    import re
    p = parsed_synthetic_dir / "parsed" / "clauses.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    all_text = " ".join(r["text"] for r in rows)
    normalised = re.sub(r"\s+", " ", all_text)
    assert "reasonable levels of dust suppression" in normalised
    assert "sufficient number of qualified safety officers" in normalised
    assert "regular intervals" in normalised


def test_parse_assigns_unique_clause_ids(parsed_synthetic_dir):
    p = parsed_synthetic_dir / "parsed" / "clauses.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    ids = [r["clause_id"] for r in rows]
    assert len(ids) == len(set(ids)), "clause_ids must be unique"


def test_parse_records_clause_metadata(parsed_synthetic_dir):
    p = parsed_synthetic_dir / "parsed" / "clauses.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    sample = rows[0]
    # Required fields
    for field in ("clause_id", "doc_id", "tender_id", "page", "text",
                  "char_count", "word_count", "table_origin"):
        assert field in sample, f"missing field: {field}"
