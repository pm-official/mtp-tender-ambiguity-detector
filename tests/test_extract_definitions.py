"""Offline tests for the definitions extractor."""

from __future__ import annotations

from pathlib import Path

from src.extract.definitions import _clean_term, extract_definitions
from src.schemas.clause import Clause


def _clause(text: str, *, clause_id="c1", section="(unsectioned)") -> Clause:
    return Clause(
        clause_id=clause_id,
        doc_id="doc.pdf",
        tender_id="T1",
        page=1,
        page_end=1,
        section_path=[section],
        clause_number=None,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
    )


def test_extract_simple_means_pattern():
    text = '"the Contractor" means the entity whose tender has been accepted by the Authority.'
    defs = extract_definitions([_clause(text)])
    assert len(defs) >= 1
    assert any("Contractor" in d.canonical_form for d in defs)


def test_extract_shall_mean_pattern():
    text = 'In these Conditions, "the Engineer" shall mean the person appointed by the Authority to administer the Contract.'
    defs = extract_definitions([_clause(text)])
    assert len(defs) >= 1
    assert any("Engineer" in d.canonical_form for d in defs)


def test_extract_paren_means_pattern():
    text = '(Contract) means the agreement entered into by the parties for the construction of the Works.'
    defs = extract_definitions([_clause(text)])
    assert len(defs) >= 1
    assert any("Contract" in d.canonical_form for d in defs)


def test_extract_nothing_when_no_definition():
    text = 'The Contractor shall execute the Works in accordance with the Specifications.'
    defs = extract_definitions([_clause(text)])
    assert defs == []


def test_extract_dedupe_within_doc():
    text = '"the Contractor" means the entity. "the Contractor" means the entity again.'
    defs = extract_definitions([_clause(text)])
    # Should dedupe — only one Contractor definition per (term, doc_id)
    contractor_defs = [d for d in defs if "Contractor" in d.canonical_form]
    assert len(contractor_defs) == 1


def test_extract_records_provenance():
    text = '"the Engineer" means the person appointed by the Authority.'
    defs = extract_definitions([_clause(text, clause_id="my_clause_42")])
    assert defs[0].defined_in_clause == "my_clause_42"
    assert defs[0].doc_id == "doc.pdf"
    assert defs[0].extraction_method == "regex"


def test_clean_term_strips_articles_and_punctuation():
    assert _clean_term('"the Contractor"') == "Contractor"
    assert _clean_term("a Specification.") == "Specification"
    assert _clean_term("THE EMPLOYER") == "EMPLOYER"


def test_extract_handles_empty_clause_list():
    assert extract_definitions([]) == []
