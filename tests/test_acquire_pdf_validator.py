"""Offline tests for src.acquire.pdf_validator."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from src.acquire.pdf_validator import inspect_pdf, quick_magic_check


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp(prefix="pdf_validator_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_quick_magic_check_rejects_non_pdf(tmp_dir):
    p = tmp_dir / "fake.pdf"
    p.write_bytes(b"<html>not a pdf</html>")
    assert quick_magic_check(p) is False


def test_quick_magic_check_accepts_pdf_header(tmp_dir):
    p = tmp_dir / "fake.pdf"
    p.write_bytes(b"%PDF-1.4\n... rest of file ...")
    assert quick_magic_check(p) is True


def test_quick_magic_check_handles_missing_file(tmp_dir):
    p = tmp_dir / "does_not_exist.pdf"
    assert quick_magic_check(p) is False


def test_inspect_pdf_rejects_non_pdf(tmp_dir):
    p = tmp_dir / "fake.pdf"
    p.write_bytes(b"this is not a pdf")
    info = inspect_pdf(p)
    assert info.is_pdf is False
    assert info.page_count == 0
    assert info.error is not None


def test_inspect_pdf_accepts_real_pdf(tmp_dir):
    """Generate a tiny real PDF with PyMuPDF, then inspect it."""
    pytest.importorskip("fitz")
    import fitz

    p = tmp_dir / "real.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World")
    doc.save(p)
    doc.close()

    info = inspect_pdf(p)
    assert info.is_pdf is True
    assert info.page_count == 1
    assert info.has_text_layer is True
    assert info.error is None
    assert info.file_size_bytes > 0
