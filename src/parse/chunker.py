"""Clause-aware chunker.

Takes a ParsedDocument (from parser.py) and produces a list of Clause
objects ready to be written to clauses.jsonl.

Two-pass strategy:
  Pass 1: Identify SECTION boundaries — high-level structural markers
          ("SECTION 5", "Annexure VIII", "Chapter 3", "Part A") that build
          the section_path that each downstream clause inherits.
  Pass 2: Inside each section, split into CLAUSES using regex on standard
          numbering patterns: 4.2.1, Clause 14, (a), (i), 10B(i), etc.

Each detected clause is bounded by the next clause-start; final clause in a
section runs to the section end.

Anything that doesn't match a clause pattern but lives between two real
clauses gets attached to the preceding clause (it's almost always a
continuation paragraph).

Tables are NOT handled in this module — they live in tables.py. The chunker
operates on flat text only, so table contents (which arrive as garbled
free-text from PyMuPDF) are filtered out by the pre-clean step and re-added
as table-row clauses by tables.py downstream.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.parse.parser import ParsedDocument
from src.schemas.clause import Clause

logger = logging.getLogger(__name__)


# ─── Patterns ───────────────────────────────────────────────────────────────
# Order matters: more specific patterns first.
# Each pattern captures the CLAUSE NUMBER as group 1.

CLAUSE_PATTERNS: list[re.Pattern[str]] = [
    # "Clause 14.2.1" / "clause 4(a)" — "Clause" or "CLAUSE" prefix
    re.compile(r"^[ \t]*Clause\s+(\d+(?:\.\d+)*[A-Z]?(?:\s*\([a-zivx]+\))?)", re.IGNORECASE | re.MULTILINE),
    # Multi-level numbering: 4.2, 4.2.1, 10B(i)
    re.compile(r"^[ \t]*(\d+(?:\.\d+)+[A-Z]?(?:\s*\([a-zivx]+\))?)\s+", re.MULTILINE),
    # Single-number-with-letter: 10B, 12A
    re.compile(r"^[ \t]*(\d+[A-Z]\.?)\s+[A-Z]", re.MULTILINE),
    # Top-level numbered: "5. " (followed by space and word char)
    re.compile(r"^[ \t]*(\d+\.)\s+[A-Za-z]", re.MULTILINE),
    # Lettered list: (a), (b), (i), (ii)
    re.compile(r"^[ \t]*\(([a-zA-Z]|[ivxIVX]+)\)\s+", re.MULTILINE),
    # Letter dot: A. B. C.
    re.compile(r"^[ \t]*([A-Z])\.\s+[A-Z]", re.MULTILINE),
]


SECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "SECTION-1: TITLE" or "SECTION 1 TITLE"
    (re.compile(r"^[ \t]*SECTION[-:\s]+(\d+|[IVXLCDM]+)\b[^\n]*", re.IGNORECASE | re.MULTILINE), "Section"),
    # "ANNEXURE-VIII" or "Annexure VIII"
    (re.compile(r"^[ \t]*ANNEXURE\s*[-–:]?\s*([IVXLCDM]+|\d+)\b[^\n]*", re.IGNORECASE | re.MULTILINE), "Annexure"),
    # "APPENDIX-A"
    (re.compile(r"^[ \t]*APPENDIX\s*[-–:]?\s*([A-Z]|\d+|[IVXLCDM]+)\b[^\n]*", re.IGNORECASE | re.MULTILINE), "Appendix"),
    # "CHAPTER 3"
    (re.compile(r"^[ \t]*CHAPTER\s+(\d+|[IVXLCDM]+)\b[^\n]*", re.IGNORECASE | re.MULTILINE), "Chapter"),
    # "PART-A" / "PART 2"
    (re.compile(r"^[ \t]*PART[-:\s]+([A-Z]|\d+)\b[^\n]*", re.IGNORECASE | re.MULTILINE), "Part"),
    # "Schedule A" or "Schedule-A"
    (re.compile(r"^[ \t]*SCHEDULE\s*[-–:]?\s*([A-Z]|\d+|[IVXLCDM]+)\b[^\n]*", re.IGNORECASE | re.MULTILINE), "Schedule"),
]


# ─── Internal data ──────────────────────────────────────────────────────────
@dataclass
class _Match:
    start: int
    end: int
    text: str
    number: str


@dataclass
class _SectionRange:
    label: str           # e.g. "Section 5", "Annexure VIII"
    start: int           # offset in concatenated text
    end: int             # offset in concatenated text


# ─── Concatenate pages with a sentinel and a page map ───────────────────────
@dataclass
class _ConcatText:
    text: str
    page_breaks: list[int]   # list of (offset_at_start_of_page) per page; len == num_pages

    def offset_to_page(self, offset: int) -> int:
        """Return 1-indexed page number for a given character offset."""
        # binary search would be faster but linear is fine for a few hundred pages
        page = 1
        for i, brk in enumerate(self.page_breaks):
            if brk <= offset:
                page = i + 1
            else:
                break
        return page


def _concat_pages(parsed: ParsedDocument) -> _ConcatText:
    """Concatenate all page texts, recording where each page starts."""
    parts: list[str] = []
    breaks: list[int] = []
    cursor = 0
    for page in parsed.pages:
        breaks.append(cursor)
        parts.append(page.text)
        cursor += len(page.text)
        # Inject a clear page break so regex matches don't bleed across pages
        # for patterns anchored to ^.
        parts.append("\n")
        cursor += 1
    return _ConcatText(text="".join(parts), page_breaks=breaks)


# ─── Section pass ───────────────────────────────────────────────────────────
def _find_sections(concat: _ConcatText) -> list[_SectionRange]:
    """Scan for top-level section markers across the concatenated text.
    Returns sections in order; first section may start after offset 0 if the
    document has front-matter."""
    matches: list[tuple[int, str]] = []   # (offset, label)
    for pattern, kind in SECTION_PATTERNS:
        for m in pattern.finditer(concat.text):
            num = m.group(1)
            label = f"{kind} {num}"
            matches.append((m.start(), label))

    if not matches:
        # No sections detected — treat entire doc as one section.
        return [_SectionRange(label="(unsectioned)", start=0, end=len(concat.text))]

    matches.sort(key=lambda x: x[0])

    # Implicit prologue: anything before the first section marker is its own
    # "Front Matter" section so we don't lose it.
    sections: list[_SectionRange] = []
    if matches[0][0] > 0:
        sections.append(_SectionRange(label="Front Matter", start=0, end=matches[0][0]))
    for i, (off, label) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(concat.text)
        sections.append(_SectionRange(label=label, start=off, end=end))
    return sections


# ─── Clause pass ────────────────────────────────────────────────────────────
def _find_clauses_in_range(text: str, base_offset: int) -> list[_Match]:
    """Find clause-numbering matches within a slice of text. Returns matches
    with offsets in the ORIGINAL concatenated text (i.e. + base_offset)."""
    raw: list[_Match] = []
    for pattern in CLAUSE_PATTERNS:
        for m in pattern.finditer(text):
            raw.append(_Match(
                start=m.start() + base_offset,
                end=m.end() + base_offset,
                text=m.group(0),
                number=m.group(1).strip(),
            ))
    # Sort by start offset; deduplicate exact-start matches (keep the one with
    # the longest match).
    raw.sort(key=lambda x: (x.start, -len(x.text)))
    deduped: list[_Match] = []
    seen_starts: set[int] = set()
    for m in raw:
        if m.start in seen_starts:
            continue
        seen_starts.add(m.start)
        deduped.append(m)
    return deduped


# ─── Public API ─────────────────────────────────────────────────────────────
def chunk_document(
    parsed: ParsedDocument,
    *,
    min_clause_chars: int = 30,
) -> list[Clause]:
    """Top-level chunker. Produces a list of Clause objects.

    `min_clause_chars` filters out tiny matches (e.g. "(a) " followed by no
    text — page artefacts).
    """
    concat = _concat_pages(parsed)
    sections = _find_sections(concat)

    clauses: list[Clause] = []
    clause_index = 0

    for section in sections:
        section_text = concat.text[section.start:section.end]
        clause_matches = _find_clauses_in_range(section_text, base_offset=section.start)

        # If a section has no clause matches, the whole section becomes one
        # synthetic "clause" so we don't drop content.
        if not clause_matches:
            body = section_text.strip()
            if len(body) >= min_clause_chars:
                page = concat.offset_to_page(section.start)
                clauses.append(_make_clause(
                    parsed=parsed,
                    clause_index=clause_index,
                    section_label=section.label,
                    clause_number=None,
                    text=body,
                    page=page,
                    page_end=concat.offset_to_page(section.end - 1),
                ))
                clause_index += 1
            continue

        # If the section starts with text BEFORE the first clause match, that
        # text is intro/preamble text. Save it as its own non-numbered clause.
        first = clause_matches[0]
        if first.start > section.start:
            preamble = concat.text[section.start:first.start].strip()
            if len(preamble) >= min_clause_chars:
                clauses.append(_make_clause(
                    parsed=parsed,
                    clause_index=clause_index,
                    section_label=section.label,
                    clause_number=None,
                    text=preamble,
                    page=concat.offset_to_page(section.start),
                    page_end=concat.offset_to_page(first.start - 1),
                ))
                clause_index += 1

        for i, match in enumerate(clause_matches):
            body_end = (
                clause_matches[i + 1].start
                if i + 1 < len(clause_matches)
                else section.end
            )
            body = concat.text[match.start:body_end].strip()
            if len(body) < min_clause_chars:
                continue
            clauses.append(_make_clause(
                parsed=parsed,
                clause_index=clause_index,
                section_label=section.label,
                clause_number=match.number,
                text=body,
                page=concat.offset_to_page(match.start),
                page_end=concat.offset_to_page(body_end - 1),
            ))
            clause_index += 1

    return clauses


def _make_clause(
    *,
    parsed: ParsedDocument,
    clause_index: int,
    section_label: str,
    clause_number: Optional[str],
    text: str,
    page: int,
    page_end: int,
) -> Clause:
    # Was this clause's first page entirely OCR-derived?
    page_obj = parsed.pages[page - 1] if 0 < page <= len(parsed.pages) else None
    ocr_was_used = bool(page_obj and page_obj.ocr_was_used)
    return Clause(
        clause_id=f"{parsed.doc_id}::{clause_index:05d}",
        doc_id=parsed.doc_id,
        tender_id=parsed.tender_id,
        page=page,
        page_end=page_end,
        bbox=None,    # bbox derivation is a refinement; v1 omits
        section_path=[section_label] if section_label else [],
        clause_number=clause_number,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
        table_origin=False,
        ocr_was_used=ocr_was_used,
        notes="",
    )
