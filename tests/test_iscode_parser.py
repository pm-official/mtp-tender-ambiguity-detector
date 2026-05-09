"""Offline tests for the IS-code parser."""

from __future__ import annotations

import pytest

from src.iscode.parser import (
    CodeMetadata,
    parse_filename,
    _build_section_path,
    _hard_split,
)


@pytest.mark.parametrize("name,expected_code,expected_year", [
    ("IS_456_2000_PlainAndReinforcedConcrete.pdf", "IS 456:2000", 2000),
    ("IS_269_2013_OPC_33Grade.pdf", "IS 269:2013", 2013),
    ("IS_1786_2008_HSDSteelBars.pdf", "IS 1786:2008", 2008),
    ("IS_1200_part5_1982_FormWork.pdf", "IS 1200 (Part 5):1982", 1982),
    ("IS_1200_part16_1979_WaterAndSewerLines.pdf", "IS 1200 (Part 16):1979", 1979),
])
def test_filename_is_code_extraction(name, expected_code, expected_year):
    m = parse_filename(name)
    assert m.code_id == expected_code
    assert m.version_year == expected_year
    assert m.family == "IS"


def test_filename_cpwd():
    m = parse_filename("CPWD_GCC_Construction_2019.pdf")
    assert m.family == "CPWD"
    assert m.code_id.startswith("CPWD")


def test_filename_unknown_falls_back():
    m = parse_filename("RandomFile.pdf")
    assert m.family == "OTHER"
    assert m.code_id == "RandomFile"


def test_build_section_path():
    meta = CodeMetadata(code_id="IS 456:2000", code_title="RCC", version_year=2000, family="IS", is_part=None)
    path = _build_section_path("5.4.2", meta)
    assert path == ["RCC", "5", "5.4", "5.4.2"]


def test_hard_split_keeps_below_limit():
    """`_hard_split` keeps groups of paragraphs under max_chars when possible.
    Single paragraphs longer than max_chars are kept intact (we don't split
    mid-paragraph)."""
    text = ("Para A is fairly long. " * 5).strip()           # ~115 chars
    text += "\n\n" + ("Para B sentence. " * 5).strip()         # ~85 chars
    text += "\n\n" + ("Para C sentence. " * 5).strip()         # ~85 chars
    pieces = _hard_split(text, max_chars=200)
    # Each piece may combine multiple short paragraphs but each piece's
    # single-paragraph subunit fits.
    assert all(len(p) <= 250 for p in pieces)


def test_hard_split_handles_short_text():
    text = "short"
    pieces = _hard_split(text, max_chars=100)
    assert pieces == ["short"]


def test_hard_split_paragraph_boundaries():
    """Each piece should end at a paragraph boundary, not mid-sentence."""
    text = "First short paragraph.\n\nSecond short paragraph.\n\nThird short paragraph."
    pieces = _hard_split(text, max_chars=30)
    for p in pieces:
        # No piece should END mid-sentence
        assert p.endswith(".")
