"""Offline tests for src.acquire.manifest."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.acquire.manifest import (
    VALID_ROLES,
    Manifest,
    TenderDocument,
    empty_manifest,
    infer_corrigendum_sequence,
    infer_role_from_filename,
    manifest_path,
    read_manifest,
    sha256_of_file,
    write_manifest,
)


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp(prefix="manifest_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ─── Filename → role inference ──────────────────────────────────────────────
@pytest.mark.parametrize("name,expected", [
    ("corrigendum_001.pdf", "corrigendum"),
    ("Corrigendum-2.pdf", "corrigendum"),
    ("addendum_3.pdf", "corrigendum"),
    ("amendment_1.pdf", "corrigendum"),
    ("NIT_2024.pdf", "nit"),
    ("notice_inviting_tender.pdf", "nit"),
    ("GCC_v1.pdf", "gcc"),
    ("general_conditions.pdf", "gcc"),
    ("special_conditions.pdf", "scc"),
    ("particular_conditions.pdf", "scc"),
    ("technical_specs.pdf", "specs"),
    ("Specifications_volume_2.pdf", "specs"),
    ("BoQ.pdf", "boq"),
    ("schedule_of_quantities.pdf", "boq"),
    ("Drawings_index.pdf", "drawings"),
    ("Annex_A.pdf", "annex"),
    ("Appendix_C.pdf", "annex"),
    ("tender_v1.pdf", "tender_v1"),
    ("random_filename.pdf", "other"),
])
def test_role_inference(name, expected):
    assert infer_role_from_filename(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("corrigendum_001.pdf", 1),
    ("corrigendum-002.pdf", 2),
    ("addendum_3.pdf", 3),
    ("corr_2024_005.pdf", 5),
    ("amendment.pdf", 0),  # no number
])
def test_corrigendum_sequence(name, expected):
    assert infer_corrigendum_sequence(name) == expected


# ─── Manifest round-trip ────────────────────────────────────────────────────
def test_manifest_round_trip_minimal(tmp_dir):
    m = empty_manifest("TEST_001")
    write_manifest(tmp_dir, m)
    loaded = read_manifest(tmp_dir)
    assert loaded.tender_id == "TEST_001"
    assert loaded.num_documents == 0
    assert loaded.num_corrigenda == 0


def test_manifest_round_trip_with_documents(tmp_dir):
    docs = [
        TenderDocument(
            filename="01_nit.pdf",
            role="nit",
            sequence=0,
            page_count=5,
            file_size_bytes=1024,
            sha256="abc",
        ),
        TenderDocument(
            filename="corrigendum_001.pdf",
            role="corrigendum",
            sequence=1,
            page_count=2,
            file_size_bytes=512,
            sha256="def",
            issuance_date=date(2026, 2, 12),
        ),
    ]
    m = Manifest(
        tender_id="TEST_002",
        issuing_authority="NHAI",
        documents=docs,
        retrieved_at=datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc),
    )
    write_manifest(tmp_dir, m)
    loaded = read_manifest(tmp_dir)

    assert loaded.tender_id == "TEST_002"
    assert loaded.issuing_authority == "NHAI"
    assert loaded.num_documents == 2
    assert loaded.num_corrigenda == 1
    assert loaded.documents[1].issuance_date == date(2026, 2, 12)


def test_manifest_corrigenda_in_order(tmp_dir):
    docs = [
        TenderDocument(filename="c2.pdf", role="corrigendum", sequence=2, page_count=1, file_size_bytes=1, sha256="2"),
        TenderDocument(filename="c1.pdf", role="corrigendum", sequence=1, page_count=1, file_size_bytes=1, sha256="1"),
        TenderDocument(filename="c3.pdf", role="corrigendum", sequence=3, page_count=1, file_size_bytes=1, sha256="3"),
        TenderDocument(filename="nit.pdf", role="nit", sequence=0, page_count=1, file_size_bytes=1, sha256="n"),
    ]
    m = Manifest(
        tender_id="ORDER_TEST",
        documents=docs,
        retrieved_at=datetime.now(timezone.utc),
    )
    in_order = m.corrigenda_in_order()
    assert [c.sequence for c in in_order] == [1, 2, 3]


def test_manifest_invalid_role_rejected(tmp_dir):
    """Pydantic should reject roles outside the literal set."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        TenderDocument(
            filename="x.pdf",
            role="not_a_real_role",  # type: ignore[arg-type]
            sequence=0,
            page_count=1,
            file_size_bytes=1,
            sha256="x",
        )


def test_read_manifest_missing_file_raises(tmp_dir):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_dir)


def test_manifest_path_is_in_tender_dir(tmp_dir):
    p = manifest_path(tmp_dir)
    assert p.parent == tmp_dir
    assert p.name == "manifest.json"


def test_manifest_serialised_dates_are_iso(tmp_dir):
    m = empty_manifest("DATE_TEST")
    m.issue_date = date(2026, 1, 28)
    write_manifest(tmp_dir, m)
    raw = json.loads(manifest_path(tmp_dir).read_text(encoding="utf-8"))
    assert raw["issue_date"] == "2026-01-28"


# ─── SHA256 helper ──────────────────────────────────────────────────────────
def test_sha256_of_file_matches_known(tmp_dir):
    p = tmp_dir / "x.bin"
    p.write_bytes(b"hello world")
    # Known SHA256 of "hello world"
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert sha256_of_file(p) == expected


# ─── VALID_ROLES coverage ───────────────────────────────────────────────────
def test_all_valid_roles_round_trip(tmp_dir):
    """Every role in VALID_ROLES should construct without error."""
    for role in VALID_ROLES:
        d = TenderDocument(
            filename=f"x_{role}.pdf",
            role=role,  # type: ignore[arg-type]
            sequence=0,
            page_count=1,
            file_size_bytes=1,
            sha256="x",
        )
        assert d.role == role
