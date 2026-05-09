"""Parse IS-code and CPWD PDFs into section-level chunks.

Heuristic-based chunker. IS codes follow a hierarchical numbering
convention (1., 1.1, 1.1.1, ...). We split at every line that begins with a
numeric section header, and group by numeric depth.

Filename convention parsing: extracts code_id, version_year, title from
filenames like:
  IS_456_2000_PlainAndReinforcedConcrete.pdf       → IS 456:2000
  IS_1200_part5_1982_FormWork.pdf                  → IS 1200 (Part 5):1982
  CPWD_Specifications_2019_Vol1_CivilWorks.pdf     → CPWD Specifications:2019 (Vol 1)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.schemas.rewrite import ISCodeChunk

logger = logging.getLogger(__name__)


# ─── Filename → metadata ────────────────────────────────────────────────────
# IS_456_2000_PlainAndReinforcedConcrete.pdf
_IS_PATTERN = re.compile(
    r"^IS_(?P<num>\d+)(?:_part(?P<part>\d+))?_(?P<year>\d{4})_(?P<title>.+)$",
    re.IGNORECASE,
)
# CPWD_Specifications_2019_Vol1_CivilWorks.pdf
_CPWD_PATTERN = re.compile(
    r"^CPWD_(?P<title>.+?)(?:_(?P<year>\d{4}))?(?:_(?P<vol>Vol\d+))?(?:_(?P<rest>.+))?$",
    re.IGNORECASE,
)


@dataclass
class CodeMetadata:
    code_id: str            # "IS 456:2000" or "CPWD Specifications:2019 Vol 1"
    code_title: str         # Human-readable
    version_year: Optional[int]
    family: str             # "IS" or "CPWD"
    is_part: Optional[int]  # for multi-part IS codes


def parse_filename(filename: str) -> CodeMetadata:
    """Extract code identity from filename. Falls back to a generic structure
    if the filename is not in the expected form."""
    stem = Path(filename).stem
    m = _IS_PATTERN.match(stem)
    if m:
        num = m.group("num")
        part = m.group("part")
        year = m.group("year")
        title = m.group("title").replace("_", " ")
        if part:
            code_id = f"IS {num} (Part {part}):{year}"
        else:
            code_id = f"IS {num}:{year}"
        return CodeMetadata(
            code_id=code_id,
            code_title=title,
            version_year=int(year) if year else None,
            family="IS",
            is_part=int(part) if part else None,
        )

    m = _CPWD_PATTERN.match(stem)
    if m:
        title = (m.group("title") or "").replace("_", " ")
        year = m.group("year")
        vol = m.group("vol")
        rest = (m.group("rest") or "").replace("_", " ")
        full_title = " ".join([t for t in (title, rest) if t]).strip()
        code_id = f"CPWD {title}"
        if year:
            code_id += f":{year}"
        if vol:
            code_id += f" {vol}"
        return CodeMetadata(
            code_id=code_id,
            code_title=full_title or title,
            version_year=int(year) if year else None,
            family="CPWD",
            is_part=None,
        )

    return CodeMetadata(
        code_id=stem,
        code_title=stem.replace("_", " "),
        version_year=None,
        family="OTHER",
        is_part=None,
    )


# ─── Section pattern (line-level) ──────────────────────────────────────────
# Matches lines like "5", "5.4", "5.4.2", "5.4.2.1" at the START of a line,
# optionally followed by a label like "5 SCOPE". Top-level integer-only
# headings (e.g. "1") are captured too because IS codes use them.
_SECTION_LINE_PATTERN = re.compile(
    r"^[ \t]*(?P<num>\d+(?:\.\d+){0,5})(?:\s+|$)",
    re.MULTILINE,
)


# ─── Top-level chunker ─────────────────────────────────────────────────────
def parse_iscode_pdf(
    path: Path,
    *,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 4000,
) -> list[ISCodeChunk]:
    """Parse one IS-code or CPWD PDF into section-level chunks.

    Strategy:
      1. PyMuPDF text extraction with page metadata.
      2. Concatenate pages with page-break markers; build offset → page map.
      3. Identify section-header lines via regex.
      4. Slice text between successive headers; one chunk per slice.
      5. Drop tiny chunks (< min_chunk_chars). Hard-split chunks longer
         than max_chunk_chars on paragraph boundaries.
    """
    import fitz  # PyMuPDF

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    meta = parse_filename(p.name)
    doc = fitz.open(p)
    try:
        # Concatenate page texts with offset → page map
        full_text_parts: list[str] = []
        page_starts: list[int] = []
        cursor = 0
        for page in doc:
            page_starts.append(cursor)
            text = page.get_text("text") or ""
            full_text_parts.append(text)
            cursor += len(text)
            full_text_parts.append("\n")
            cursor += 1
        full_text = "".join(full_text_parts)

        def offset_to_page(off: int) -> int:
            page_idx = 0
            for i, s in enumerate(page_starts):
                if s <= off:
                    page_idx = i
                else:
                    break
            return page_idx + 1   # 1-indexed

        # Find section headers
        section_matches = list(_SECTION_LINE_PATTERN.finditer(full_text))
        if not section_matches:
            # No structure detected — return the whole doc as one chunk
            chunks = [_make_chunk(
                meta=meta, source_file=p.name,
                section="0",
                section_path=[meta.code_title],
                page=1, text=full_text.strip(),
                index=0,
            )]
            return [c for c in chunks if c.char_count >= min_chunk_chars]

        # Build chunks between successive matches
        chunks: list[ISCodeChunk] = []
        chunk_index = 0
        for i, m in enumerate(section_matches):
            start = m.start()
            end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(full_text)
            body = full_text[start:end].strip()
            if len(body) < min_chunk_chars:
                continue
            if len(body) > max_chunk_chars:
                # Split on paragraph boundaries (double-newline)
                pieces = _hard_split(body, max_chunk_chars)
                for piece in pieces:
                    if len(piece) < min_chunk_chars:
                        continue
                    chunks.append(_make_chunk(
                        meta=meta, source_file=p.name,
                        section=m.group("num"),
                        section_path=_build_section_path(m.group("num"), meta),
                        page=offset_to_page(start),
                        text=piece,
                        index=chunk_index,
                    ))
                    chunk_index += 1
            else:
                chunks.append(_make_chunk(
                    meta=meta, source_file=p.name,
                    section=m.group("num"),
                    section_path=_build_section_path(m.group("num"), meta),
                    page=offset_to_page(start),
                    text=body,
                    index=chunk_index,
                ))
                chunk_index += 1

        return chunks
    finally:
        doc.close()


def _make_chunk(
    *, meta: CodeMetadata, source_file: str, section: str,
    section_path: list[str], page: int, text: str, index: int,
) -> ISCodeChunk:
    safe_code = meta.code_id.replace(" ", "_").replace(":", "_").replace("(", "").replace(")", "")
    chunk_id = f"{safe_code}::s::{section}::{index:04d}"
    return ISCodeChunk(
        chunk_id=chunk_id,
        code_id=meta.code_id,
        code_title=meta.code_title,
        version_year=meta.version_year,
        section=section,
        section_path=section_path,
        page=page,
        text=text,
        char_count=len(text),
        source_file=source_file,
    )


def _build_section_path(section: str, meta: CodeMetadata) -> list[str]:
    """Build a hierarchical breadcrumb: [code_title, parent_section, ..., section].
    For section "5.4.2" produces [code_title, "5", "5.4", "5.4.2"]."""
    parts = section.split(".")
    out = [meta.code_title]
    for depth in range(1, len(parts) + 1):
        out.append(".".join(parts[:depth]))
    return out


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split a long text at paragraph boundaries to fit max_chars."""
    pieces: list[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            pieces.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para) if current else para
    if current.strip():
        pieces.append(current.strip())
    return pieces
