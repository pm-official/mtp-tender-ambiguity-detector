"""Pydantic schemas for parsed document structure.

The pipeline's primary unit of work is the `Clause` — a single legally-
meaningful chunk of a tender document, with enough metadata that any later
stage (BCT, retrieval, rewrite, audit) can trace back to its provenance on
the original page.

A clauses.jsonl file is one Clause per line, written as JSON.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Bounding box on the page in PyMuPDF point coordinates (origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float


class Clause(BaseModel):
    """One logically-coherent chunk of a tender document."""

    # Identity
    clause_id: str                          # unique within (doc_id) — usually "<doc_id>::<index>"
    doc_id: str                             # source PDF filename
    tender_id: str                          # parent tender folder name

    # Position
    page: int                               # 1-indexed first page where the clause starts
    page_end: int                           # last page (>= page) — clauses can span pages
    bbox: Optional[BBox] = None             # bounding box on the first page (None for OCR / table rows)

    # Structural location (best-effort)
    section_path: list[str] = Field(default_factory=list)
    clause_number: Optional[str] = None     # e.g. "5.2.1" or "(iii)" or "Clause 14"

    # Content
    text: str
    char_count: int = 0
    word_count: int = 0

    # Provenance flags
    table_origin: bool = False              # True if the clause is a linearised table row
    ocr_was_used: bool = False              # True if Tesseract OCR generated this text

    # Free-form notes (parser warnings, e.g. "table linearised", "header detected and stripped")
    notes: str = ""


class PageInfo(BaseModel):
    """Per-page metadata captured during parsing."""

    page_number: int                        # 1-indexed
    char_count: int = 0
    block_count: int = 0
    has_text_layer: bool = True             # False ⇒ scanned page; OCR was attempted
    ocr_was_used: bool = False
    width_points: float = 0.0
    height_points: float = 0.0


class DocumentTree(BaseModel):
    """Per-document parse output. Written as document_tree.json next to clauses.jsonl."""

    doc_id: str
    tender_id: str
    page_count: int
    pages: list[PageInfo] = Field(default_factory=list)
    clause_count: int = 0
    table_clause_count: int = 0
    ocr_pages: list[int] = Field(default_factory=list)
    notes: str = ""
