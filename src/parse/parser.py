"""PyMuPDF-based PDF parser with OCR fallback.

Output of this stage is a ParsedDocument: per-page text + layout + page-level
metadata. The chunker (chunker.py) consumes ParsedDocument and emits Clauses.

Design choices:
  * Pages are extracted with PyMuPDF's `page.get_text("text")` — preserves
    reading order and is fast.
  * If a page has empty text (or tiny text — could be just page numbers),
    we attempt OCR via Tesseract. OCR is OPTIONAL — if pytesseract or
    Tesseract binary is missing, we log a warning and skip OCR for that
    page rather than failing.
  * We DO NOT yet split into clauses here. That happens in chunker.py so
    the two concerns stay testable in isolation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.schemas.clause import PageInfo

logger = logging.getLogger(__name__)


# Pages with this little text (after stripping whitespace) trigger OCR.
# Empirically, even an empty cover-page sometimes has a header line of ~30 chars,
# so the threshold is conservative.
OCR_TRIGGER_CHAR_THRESHOLD = 30


@dataclass
class ParsedPage:
    page_number: int                # 1-indexed
    text: str
    char_count: int
    block_count: int
    width_points: float
    height_points: float
    has_text_layer: bool
    ocr_was_used: bool
    blocks: list[dict] = field(default_factory=list)   # PyMuPDF text blocks


@dataclass
class ParsedDocument:
    doc_id: str                     # filename
    tender_id: str
    path: Path
    page_count: int
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def ocr_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.ocr_was_used]

    def page_info_list(self) -> list[PageInfo]:
        return [
            PageInfo(
                page_number=p.page_number,
                char_count=p.char_count,
                block_count=p.block_count,
                has_text_layer=p.has_text_layer,
                ocr_was_used=p.ocr_was_used,
                width_points=p.width_points,
                height_points=p.height_points,
            )
            for p in self.pages
        ]


# ─── Page-text cleanup ──────────────────────────────────────────────────────
_REPEATED_HEADER_FOOTER_RX = re.compile(
    r"^(Page\s+\d+\s+of\s+\d+|Page\s+\d+|\d+\s*/\s*\d+|\d+)$",
    re.IGNORECASE,
)


def _strip_obvious_header_footer_lines(text: str) -> str:
    """Remove standalone page-number and "Page X of Y" lines.
    Keeps everything else untouched."""
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        if _REPEATED_HEADER_FOOTER_RX.match(stripped):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


# ─── OCR (optional) ─────────────────────────────────────────────────────────
def _ocr_page(page) -> Optional[str]:
    """Render the page to an image and extract text via OCR.

    Tries in order:
      1. Local Tesseract via pytesseract (fast, offline, free)
      2. Gemini Flash Vision API (slower, costs ~₹0.10 per page, but works
         on Streamlit Cloud and other environments without system deps)

    Returns the extracted text, or None if both routes fail.
    """
    # ── Route 1: Tesseract ───────────────────────────────────────────────
    tess_text = _ocr_via_tesseract(page)
    if tess_text and tess_text.strip():
        return tess_text

    # ── Route 2: Gemini Vision fallback ──────────────────────────────────
    gem_text = _ocr_via_gemini(page)
    if gem_text and gem_text.strip():
        return gem_text

    return None


def _ocr_via_tesseract(page) -> Optional[str]:
    """Local Tesseract OCR. Returns None if unavailable."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None

    try:
        pix = page.get_pixmap(dpi=200, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(img, lang="eng")
    except pytesseract.TesseractNotFoundError:
        logger.info(
            "Tesseract binary not on PATH; falling back to Gemini Vision OCR. "
            "Install Tesseract for faster + free OCR."
        )
        return None
    except Exception as e:
        logger.warning("Tesseract OCR failed for page %d: %s", page.number + 1, e)
        return None


def _ocr_via_gemini(page) -> Optional[str]:
    """Render the page to a JPEG and extract text with Gemini Flash Vision.

    Cost is roughly ~₹0.10 per page (Flash on a ~1MB image). Used only as a
    fallback when Tesseract is unavailable, so a typical run on a text-PDF
    tender pays nothing here.
    """
    try:
        # Render at 150 DPI — high enough for clean OCR, small enough to
        # keep the upload payload reasonable.
        pix = page.get_pixmap(dpi=150, alpha=False)
        jpeg_bytes = pix.tobytes(output="jpeg", jpg_quality=85)
    except Exception as e:
        logger.warning("Page-render to JPEG failed for page %d: %s", page.number + 1, e)
        return None

    prompt = (
        "Extract all text from this PDF page exactly as it appears, "
        "preserving line breaks and clause numbering. Do not paraphrase "
        "or summarise. If the page has no text, respond with the empty string."
    )

    # Use the fallback client when configured, so Vision OCR fails over
    # from the primary Gemini key to the secondary on quota exhaustion.
    try:
        from src.llm.gemini_client import get_default_client
        client = get_default_client()
    except Exception as e:
        logger.warning("Could not init LLM client for Vision OCR: %s", e)
        return None

    # Multi-backend wrapper has a dedicated vision_extract method
    if hasattr(client, "vision_extract"):
        try:
            text = client.vision_extract(jpeg_bytes, prompt, mime_type="image/jpeg")
            if text:
                logger.info(
                    "Gemini Vision OCR'd page %d (%d chars).",
                    page.number + 1, len(text),
                )
            return text or None
        except Exception as e:
            logger.warning(
                "Vision OCR failed for page %d: %s", page.number + 1, e,
            )
            return None

    # Plain GeminiClient fallback
    try:
        from google.genai import types as gtypes
        resp = client._client.models.generate_content(
            model=client.default_flash,
            contents=[
                gtypes.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )
        text = (resp.text or "").strip()
        if text:
            logger.info(
                "Gemini Vision OCR'd page %d (%d chars).",
                page.number + 1, len(text),
            )
        return text or None
    except Exception as e:
        logger.warning(
            "Gemini Vision OCR failed for page %d: %s", page.number + 1, e
        )
        return None


# ─── Main entry point ───────────────────────────────────────────────────────
def parse_pdf(
    path: Path,
    *,
    tender_id: str,
    doc_id: Optional[str] = None,
    enable_ocr: bool = True,
    strip_obvious_headers: bool = True,
) -> ParsedDocument:
    """Parse a single PDF.

    Parameters
    ----------
    path : Path to the PDF file.
    tender_id : The parent tender folder name.
    doc_id : Defaults to path.name.
    enable_ocr : If True, run Tesseract OCR on pages with no text layer.
    strip_obvious_headers : If True, drop standalone "Page X of Y" lines.
    """
    import fitz  # PyMuPDF

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    doc_id = doc_id or p.name
    doc = fitz.open(p)
    try:
        pages: list[ParsedPage] = []
        for page in doc:
            page_number = page.number + 1   # 1-indexed for human-readability

            # Get text and basic structure
            try:
                text = page.get_text("text") or ""
            except Exception as e:
                logger.warning("get_text failed on %s page %d: %s", doc_id, page_number, e)
                text = ""

            blocks: list[dict] = []
            try:
                blocks = page.get_text("dict").get("blocks", []) or []
            except Exception as e:
                logger.debug("get_text('dict') failed on %s page %d: %s", doc_id, page_number, e)

            has_text_layer = len(text.strip()) >= OCR_TRIGGER_CHAR_THRESHOLD
            ocr_was_used = False

            if not has_text_layer and enable_ocr:
                ocr_text = _ocr_page(page)
                if ocr_text and len(ocr_text.strip()) >= OCR_TRIGGER_CHAR_THRESHOLD:
                    text = ocr_text
                    ocr_was_used = True

            if strip_obvious_headers:
                text = _strip_obvious_header_footer_lines(text)

            rect = page.rect
            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=text,
                    char_count=len(text),
                    block_count=len(blocks),
                    width_points=rect.width,
                    height_points=rect.height,
                    has_text_layer=has_text_layer,
                    ocr_was_used=ocr_was_used,
                    blocks=blocks,
                )
            )

        return ParsedDocument(
            doc_id=doc_id,
            tender_id=tender_id,
            path=p,
            page_count=len(pages),
            pages=pages,
        )
    finally:
        doc.close()
