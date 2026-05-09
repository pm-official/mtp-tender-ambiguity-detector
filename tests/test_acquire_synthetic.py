"""End-to-end test of the synthetic tender generator."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from src.acquire.manifest import read_manifest
from src.acquire.pdf_validator import inspect_pdf
from src.acquire.synthetic import generate_synthetic_tender


@pytest.fixture
def tmp_corpus():
    d = Path(tempfile.mkdtemp(prefix="syn_tender_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_synthetic_tender_complete(tmp_corpus):
    """Generate a synthetic tender, verify all 7 PDFs + manifest exist and validate."""
    folder = generate_synthetic_tender(tmp_corpus, tender_id="SYN_TEST_001")

    # Files
    assert (folder / "01_nit.pdf").exists()
    assert (folder / "02_gcc.pdf").exists()
    assert (folder / "03_scc.pdf").exists()
    assert (folder / "04_specs.pdf").exists()
    assert (folder / "05_boq.pdf").exists()
    assert (folder / "corrigendum_001.pdf").exists()
    assert (folder / "corrigendum_002.pdf").exists()
    assert (folder / "manifest.json").exists()

    # Each PDF is real and has > 0 pages
    for pdf in folder.glob("*.pdf"):
        info = inspect_pdf(pdf)
        assert info.is_pdf, f"{pdf.name} is not a valid PDF: {info.error}"
        assert info.page_count >= 1
        assert info.has_text_layer, f"{pdf.name} should have a text layer"

    # Manifest reads cleanly
    m = read_manifest(folder)
    assert m.tender_id == "SYN_TEST_001"
    assert m.num_documents == 7
    assert m.num_corrigenda == 2

    # Corrigenda are in sequence order with their issuance dates
    corr = m.corrigenda_in_order()
    assert corr[0].sequence == 1
    assert corr[1].sequence == 2
    assert corr[0].issuance_date is not None
    assert corr[1].issuance_date is not None
    assert corr[1].issuance_date > corr[0].issuance_date

    # SHA256 is non-empty for every doc
    for d in m.documents:
        assert len(d.sha256) == 64

    # Roles are sensible
    role_counts = {r: 0 for r in {"nit", "gcc", "scc", "specs", "boq", "corrigendum"}}
    for d in m.documents:
        if d.role in role_counts:
            role_counts[d.role] += 1
    assert role_counts["corrigendum"] == 2
    # The other 5 documents may map to specific roles or 'other' depending on
    # filename heuristic; at minimum, NIT, GCC, SCC, Specs, BoQ should each
    # be tagged correctly.
    assert role_counts["nit"] >= 1
    assert role_counts["gcc"] >= 1
    assert role_counts["scc"] >= 1
    assert role_counts["specs"] >= 1
    assert role_counts["boq"] >= 1


def test_synthetic_tender_refuses_to_overwrite(tmp_corpus):
    """Calling generate twice with same ID should error unless overwrite=True."""
    generate_synthetic_tender(tmp_corpus, tender_id="DOUBLE_TEST")
    with pytest.raises(FileExistsError):
        generate_synthetic_tender(tmp_corpus, tender_id="DOUBLE_TEST")
    # With overwrite=True, no error.
    generate_synthetic_tender(tmp_corpus, tender_id="DOUBLE_TEST", overwrite=True)
