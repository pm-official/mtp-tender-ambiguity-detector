"""Offline tests for the references extractor."""

from __future__ import annotations

from src.extract.references import build_reference_graph, extract_references
from src.schemas.clause import Clause


def _clause(text: str, *, clause_id="c1", clause_number=None) -> Clause:
    return Clause(
        clause_id=clause_id,
        doc_id="doc.pdf",
        tender_id="T1",
        page=1,
        page_end=1,
        section_path=[],
        clause_number=clause_number,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
    )


def test_extract_internal_clause_reference():
    refs = extract_references([_clause("As provided in Clause 4.6 of these Conditions.")])
    assert any(r.target_kind == "clause" and "4.6" in r.target_label for r in refs)


def test_extract_section_reference():
    refs = extract_references([_clause("See Section 5.")])
    assert any(r.target_kind == "section" for r in refs)


def test_extract_annex_reference():
    refs = extract_references([_clause("As per Annexure VIII.")])
    assert any(r.target_kind == "annex" and "VIII" in r.target_label for r in refs)


def test_extract_appendix_reference():
    refs = extract_references([_clause("Appendix C is hereby attached.")])
    assert any(r.target_kind == "appendix" and "C" in r.target_label for r in refs)


def test_extract_is_code_reference():
    refs = extract_references([_clause("conform to IS 456:2000 Section 9.")])
    assert any(r.target_kind == "standard" and "456" in r.raw_text for r in refs)


def test_extract_morth_reference():
    refs = extract_references([_clause("complying with MoRTH 5.0 specifications.")])
    assert any(r.target_kind == "standard" and "MoRTH" in r.raw_text for r in refs)


def test_extract_no_references_in_plain_prose():
    text = "The Contractor shall maintain workmanship of high quality."
    refs = extract_references([_clause(text)])
    assert len(refs) == 0


def test_build_graph_resolves_internal_clause():
    """A reference 'Clause 4.6' from clause 'A' should resolve to clause 'B'
    if B has clause_number=4.6."""
    a = _clause("As per Clause 4.6.", clause_id="A", clause_number="3.1")
    b = _clause("This is the resolving clause body for clause 4.6 here.",
                clause_id="B", clause_number="4.6")
    refs = extract_references([a, b])
    graph, refs_resolved = build_reference_graph([a, b], refs)

    # The reference from A → 4.6 should resolve to B
    resolved = [r for r in refs_resolved if r.resolved]
    assert any(r.target_id == "B" for r in resolved)
    assert graph.has_edge("A", "B")


def test_build_graph_external_standard_kept_as_external_node():
    a = _clause("As per IS 456:2000.", clause_id="A")
    refs = extract_references([a])
    graph, refs_resolved = build_reference_graph([a], refs)
    # No internal resolution; should add an external node
    external_nodes = [n for n, d in graph.nodes(data=True) if d.get("external")]
    assert len(external_nodes) >= 1


def test_extract_handles_empty_clause_list():
    assert extract_references([]) == []
