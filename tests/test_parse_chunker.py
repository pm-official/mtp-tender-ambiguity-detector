"""Offline tests for the clause-aware chunker."""

from __future__ import annotations

from src.parse.chunker import (
    CLAUSE_PATTERNS,
    SECTION_PATTERNS,
    _ConcatText,
    _find_clauses_in_range,
    _find_sections,
    chunk_document,
)
from src.parse.parser import ParsedDocument, ParsedPage


# ─── Helpers ────────────────────────────────────────────────────────────────
def _make_parsed(text_per_page: list[str], doc_id: str = "test.pdf",
                  tender_id: str = "TEST") -> ParsedDocument:
    pages = [
        ParsedPage(
            page_number=i + 1,
            text=text,
            char_count=len(text),
            block_count=0,
            width_points=595.0,
            height_points=842.0,
            has_text_layer=True,
            ocr_was_used=False,
            blocks=[],
        )
        for i, text in enumerate(text_per_page)
    ]
    from pathlib import Path

    return ParsedDocument(
        doc_id=doc_id,
        tender_id=tender_id,
        path=Path(f"/tmp/{doc_id}"),
        page_count=len(pages),
        pages=pages,
    )


# ─── Section detection ──────────────────────────────────────────────────────
def test_section_pattern_matches_section_dash_number():
    pat, kind = SECTION_PATTERNS[0]
    m = pat.search("\nSECTION-1: Notice Inviting Tender\n")
    assert m is not None
    assert m.group(1) == "1"
    assert kind == "Section"


def test_section_pattern_matches_annexure_roman():
    pat, kind = SECTION_PATTERNS[1]
    m = pat.search("\nAnnexure-VIII\n")
    assert m is not None
    assert m.group(1) == "VIII"


def test_section_pattern_matches_part_letter():
    pat, kind = SECTION_PATTERNS[4]
    m = pat.search("\nPART-A\n")
    assert m is not None
    assert m.group(1) == "A"


def test_find_sections_returns_implicit_front_matter():
    text = "Some preamble text here.\n\nSECTION-1\n\nReal content\n\nSECTION-2\n\nMore\n"
    concat = _ConcatText(text=text, page_breaks=[0])
    sections = _find_sections(concat)
    assert sections[0].label == "Front Matter"
    assert any(s.label == "Section 1" for s in sections)
    assert any(s.label == "Section 2" for s in sections)


def test_find_sections_empty_doc():
    concat = _ConcatText(text="just one paragraph", page_breaks=[0])
    sections = _find_sections(concat)
    assert len(sections) == 1
    assert sections[0].label == "(unsectioned)"


# ─── Clause patterns ────────────────────────────────────────────────────────
def test_clause_patterns_match_multilevel_numbering():
    text = "\n4.2.1  The Contractor shall do something."
    matches = []
    for pat in CLAUSE_PATTERNS:
        for m in pat.finditer(text):
            matches.append(m.group(1))
    assert "4.2.1" in matches


def test_clause_patterns_match_clause_keyword():
    text = "\nClause 14.2 The Contractor shall..."
    matches = []
    for pat in CLAUSE_PATTERNS:
        for m in pat.finditer(text):
            matches.append(m.group(1))
    assert any("14.2" in m for m in matches)


def test_clause_patterns_match_lettered_list():
    text = "\n(a) First item. \n(b) Second item.\n(c) Third item."
    matches = []
    for pat in CLAUSE_PATTERNS:
        for m in pat.finditer(text):
            matches.append(m.group(1))
    assert "a" in matches
    assert "b" in matches
    assert "c" in matches


def test_clause_patterns_match_roman_numerals():
    text = "\n(i) First. \n(ii) Second. \n(iii) Third."
    matches = []
    for pat in CLAUSE_PATTERNS:
        for m in pat.finditer(text):
            matches.append(m.group(1))
    assert any(m.lower() == "i" for m in matches)


def test_find_clauses_in_range_dedupes_overlaps():
    """Overlapping pattern matches at the same start offset should dedupe."""
    text = "\n4.2  The Contractor shall do something specific."
    out = _find_clauses_in_range(text, base_offset=0)
    # only one match should remain at offset of "4.2"
    starts = [m.start for m in out]
    assert len(starts) == len(set(starts))


# ─── End-to-end chunking ────────────────────────────────────────────────────
def test_chunk_simple_document():
    text = """SECTION-1: Test Section

1.1  First clause text. The Contractor shall do X.

1.2  Second clause text. The Contractor shall do Y.

1.3  Third clause with more substantive content here for testing.
"""
    parsed = _make_parsed([text])
    clauses = chunk_document(parsed)
    nums = [c.clause_number for c in clauses if c.clause_number]
    assert "1.1" in nums
    assert "1.2" in nums
    assert "1.3" in nums
    # Section path includes Section 1
    assert any("Section 1" in c.section_path for c in clauses)


def test_chunk_filters_tiny_clauses():
    text = "1.1 hi\n1.2 also short\n1.3 a much longer clause body that should pass the minimum char filter easily.\n"
    parsed = _make_parsed([text])
    clauses = chunk_document(parsed, min_clause_chars=30)
    # 1.1 and 1.2 are under 30 chars; 1.3 is over.
    nums = {c.clause_number for c in clauses}
    assert "1.3" in nums


def test_chunk_preserves_page_metadata():
    page1 = "1.1 First clause body that is sufficiently long to pass the filter for inclusion.\n"
    page2 = "1.2 Second clause body that is also sufficiently long for inclusion in the output.\n"
    parsed = _make_parsed([page1, page2])
    clauses = chunk_document(parsed)
    # 1.1 should be on page 1, 1.2 on page 2
    by_num = {c.clause_number: c for c in clauses if c.clause_number}
    assert by_num["1.1"].page == 1
    assert by_num["1.2"].page == 2


def test_chunk_assigns_clause_ids_uniquely():
    text = """1.1 First clause here that is plenty long enough.
1.2 Second clause here that is also long enough.
1.3 Third clause here that is plenty long enough.
"""
    parsed = _make_parsed([text])
    clauses = chunk_document(parsed)
    ids = [c.clause_id for c in clauses]
    assert len(ids) == len(set(ids)), "clause_ids must be unique"


def test_chunk_handles_no_clause_patterns():
    text = "This document has no clause numbering at all. Just prose. " * 5
    parsed = _make_parsed([text])
    clauses = chunk_document(parsed)
    # We still get one synthetic clause (the whole doc)
    assert len(clauses) >= 1
    assert clauses[0].clause_number is None
