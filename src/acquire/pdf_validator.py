"""PDF validation utilities.

We trust PyMuPDF (fitz) as the source of truth for "is this a real PDF and
how many pages does it have". This module also captures a few summary
statistics that future stages can reuse without re-opening the PDF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# PDF magic number (first 4 bytes are "%PDF").
PDF_MAGIC = b"%PDF"


@dataclass
class PDFInfo:
    path: Path
    is_pdf: bool
    page_count: int
    file_size_bytes: int
    is_encrypted: bool = False
    has_text_layer: bool = True
    error: str | None = None


def quick_magic_check(path: Path) -> bool:
    """Cheap "does the file start with %PDF?" check. Catches obvious download
    failures (empty files, HTML error pages, ZIP downloads, etc.) without
    requiring PyMuPDF."""
    try:
        with Path(path).open("rb") as f:
            head = f.read(4)
        return head == PDF_MAGIC
    except OSError:
        return False


def inspect_pdf(path: Path, sample_text_layer: bool = True) -> PDFInfo:
    """Open a PDF with PyMuPDF and return summary information. Never raises;
    captures any error in `error` and returns is_pdf=False."""
    p = Path(path)
    file_size = p.stat().st_size if p.exists() else 0

    if not quick_magic_check(p):
        return PDFInfo(
            path=p,
            is_pdf=False,
            page_count=0,
            file_size_bytes=file_size,
            error="File does not start with %PDF magic bytes (likely corrupt or wrong type).",
        )

    try:
        import fitz  # PyMuPDF
    except ImportError:
        # Without PyMuPDF we can still report magic-check + size.
        return PDFInfo(
            path=p,
            is_pdf=True,
            page_count=0,
            file_size_bytes=file_size,
            error="PyMuPDF not installed; magic check passed but page count unknown.",
        )

    try:
        doc = fitz.open(p)
    except Exception as e:
        return PDFInfo(
            path=p,
            is_pdf=False,
            page_count=0,
            file_size_bytes=file_size,
            error=f"PyMuPDF failed to open: {e}",
        )

    try:
        is_encrypted = bool(doc.needs_pass)
        page_count = len(doc)

        has_text = True
        if sample_text_layer and page_count > 0:
            # Sample first up-to-3 pages; if all are empty-string we flag the
            # PDF as image-only (downstream chunker should run OCR).
            sampled = 0
            empty = 0
            for i in range(min(3, page_count)):
                txt = doc[i].get_text("text") or ""
                if not txt.strip():
                    empty += 1
                sampled += 1
            has_text = empty < sampled  # at least one page had text

        return PDFInfo(
            path=p,
            is_pdf=True,
            page_count=page_count,
            file_size_bytes=file_size,
            is_encrypted=is_encrypted,
            has_text_layer=has_text,
        )
    finally:
        doc.close()
