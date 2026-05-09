"""Table extraction + linearisation.

PyMuPDF 1.23+ has `page.find_tables()` which returns detected tables with
cells. For each table row, we produce a synthetic Clause whose text is
"<col1_header>: <col1_value>; <col2_header>: <col2_value>; ...".

These synthetic table-row clauses have `table_origin = True` so downstream
stages can treat them differently if needed. For BCT, table rows are
high-value: they often contain numerical commitments (BoQ, schedules of
rates, performance tables) that vagueness detection should target directly.

The first row of each detected table is usually the header. If `find_tables`
gives us a `header` attribute, we use it; otherwise we fall back to "Col 1",
"Col 2", ...
"""

from __future__ import annotations

import logging
from typing import Optional

from src.parse.parser import ParsedDocument
from src.schemas.clause import Clause

logger = logging.getLogger(__name__)


def extract_tables_as_clauses(
    parsed: ParsedDocument,
    *,
    starting_clause_index: int = 0,
    min_row_chars: int = 30,
) -> list[Clause]:
    """Run table detection on every page, linearise rows, return as Clause list.

    Note: this re-opens the PDF because PyMuPDF's table detector needs a live
    Page object, not a cached `dict`. The cost is small (PDFs are mmap'd).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed; skipping table extraction.")
        return []

    if not hasattr(fitz.Page, "find_tables"):
        logger.warning(
            "Installed PyMuPDF version lacks find_tables (need >= 1.23). "
            "Skipping table extraction."
        )
        return []

    out: list[Clause] = []
    cursor = starting_clause_index

    doc = fitz.open(parsed.path)
    try:
        for page_idx, page in enumerate(doc):
            page_number = page_idx + 1
            try:
                tables_finder = page.find_tables()
            except Exception as e:
                logger.debug("find_tables() failed on page %d: %s", page_number, e)
                continue

            tables = getattr(tables_finder, "tables", []) or []
            for tbl_idx, table in enumerate(tables):
                rows = _extract_rows(table)
                if not rows:
                    continue

                header = rows[0]
                body_rows = rows[1:]
                if not body_rows:
                    continue

                for row in body_rows:
                    text = _linearise_row(header, row)
                    if len(text) < min_row_chars:
                        continue
                    out.append(Clause(
                        clause_id=f"{parsed.doc_id}::table_{page_number:03d}_{tbl_idx:02d}_r{cursor:05d}",
                        doc_id=parsed.doc_id,
                        tender_id=parsed.tender_id,
                        page=page_number,
                        page_end=page_number,
                        bbox=None,
                        section_path=[f"Table p{page_number}#{tbl_idx + 1}"],
                        clause_number=None,
                        text=text,
                        char_count=len(text),
                        word_count=len(text.split()),
                        table_origin=True,
                        ocr_was_used=False,
                        notes="linearised table row",
                    ))
                    cursor += 1
    finally:
        doc.close()

    return out


def _extract_rows(table) -> list[list[str]]:
    """Return rows as list-of-list-of-strings. Robust to None cells."""
    try:
        raw = table.extract()
    except Exception:
        return []
    rows: list[list[str]] = []
    for r in raw or []:
        cleaned = [(c or "").strip().replace("\n", " ") for c in r]
        if any(cleaned):
            rows.append(cleaned)
    return rows


def _linearise_row(header: list[str], row: list[str]) -> str:
    """Render a row as 'Col1: val1; Col2: val2; ...'. Pads or truncates if
    header and row lengths differ."""
    n = max(len(header), len(row))
    parts: list[str] = []
    for i in range(n):
        h = header[i] if i < len(header) else f"Col{i + 1}"
        v = row[i] if i < len(row) else ""
        h = h.strip() or f"Col{i + 1}"
        v = v.strip()
        if not v:
            continue
        parts.append(f"{h}: {v}")
    return "; ".join(parts)
