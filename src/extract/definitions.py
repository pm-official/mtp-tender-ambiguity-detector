"""Defined-term extractor.

Two-pass strategy:
  Pass 1 — Regex on common patterns. Catches the bulk cheaply:
    "X" means "Y"
    "X" shall mean "Y"
    "X" is defined as "Y"
    By "X" is meant "Y"
    (X) means Y          <-- Indian contracts use this a lot
  Pass 2 — Optional LLM fallback (Gemini Flash, batched) on clauses inside
           any "Definitions" section that pass-1 missed. Cost is a few
           batched calls per tender.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field

from src.schemas.clause import Clause
from src.schemas.structures import Definition

logger = logging.getLogger(__name__)


# ─── Regex patterns for "X means Y" ────────────────────────────────────────
# Each pattern captures (term, definition_text). Term is in group "term",
# definition is the rest of the sentence up to a clause delimiter.

# Pattern 1: "the term X" or '"X"' followed by means/shall mean/refers to
_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"""
        (?:^|[\s\.;])                    # boundary
        (?:["'`“‘]|the\s+(?:expression|term|word)\s+)?
        (?P<term>[A-Z][A-Za-z\-\s/]{1,80}?)
        (?:["'”’])?
        \s+
        (?:shall\s+mean|means|refers\s+to|is\s+defined\s+as|is\s+meant\s+to\s+(?:include|mean))
        \s+
        (?P<definition>.+?)
        (?=[.;]|$)
        """,
        re.VERBOSE | re.IGNORECASE | re.DOTALL,
    ),
    # "(X) means Y" — common in Indian Special Conditions
    re.compile(
        r"""
        \(\s*(?P<term>[A-Za-z][A-Za-z\-\s/]{1,80}?)\s*\)
        \s+
        (?:shall\s+mean|means)
        \s+
        (?P<definition>.+?)
        (?=[.;]|$)
        """,
        re.VERBOSE | re.IGNORECASE | re.DOTALL,
    ),
]


# Heuristic: is this clause inside a Definitions / Glossary section?
def _looks_like_definitions_section(clause: Clause) -> bool:
    sp = " > ".join(clause.section_path).lower()
    if "definition" in sp or "glossary" in sp or "interpretation" in sp:
        return True
    text = clause.text[:200].lower()
    if "definition" in text or "interpretation" in text:
        return True
    return False


def _clean_term(t: str) -> str:
    """Normalise a candidate defined-term string."""
    t = re.sub(r"\s+", " ", t).strip()
    # strip surrounding quotes / punctuation FIRST so the article-strip works
    t = t.strip(" .,:;\"'`“”‘’")
    # strip leading articles
    t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.IGNORECASE)
    # final whitespace tidy
    t = t.strip()
    return t


def _clean_definition_body(body: str) -> str:
    body = re.sub(r"\s+", " ", body).strip()
    body = body.strip(" .,:;\"'`“”‘’")
    return body


# ─── Public API ─────────────────────────────────────────────────────────────
def extract_definitions(
    clauses: Iterable[Clause],
    *,
    enable_llm_fallback: bool = False,
    llm_only_in_definitions_sections: bool = True,
) -> list[Definition]:
    """Extract every defined term from a list of clauses.

    Parameters
    ----------
    clauses : iterable of Clause
        Output of the chunker.
    enable_llm_fallback : bool
        If True, send clauses inside Definitions sections that produced no
        regex matches to Gemini Flash for LLM extraction. Off by default
        for cost reasons.
    """
    found: list[Definition] = []
    seen_canonical: set[tuple[str, str]] = set()  # (canonical_form_lower, doc_id)

    clauses = list(clauses)

    # Pass 1 — regex
    for clause in clauses:
        for pat in _PATTERNS:
            for m in pat.finditer(clause.text):
                term = _clean_term(m.group("term"))
                body = _clean_definition_body(m.group("definition"))
                if not term or not body or len(body) < 10:
                    continue
                # Reject obviously-non-term matches: too long, all lowercase, etc.
                if len(term) > 80 or term.count(" ") > 6:
                    continue
                key = (term.lower(), clause.doc_id)
                if key in seen_canonical:
                    continue
                seen_canonical.add(key)
                found.append(Definition(
                    canonical_form=term,
                    definition_text=body,
                    defined_in_clause=clause.clause_id,
                    doc_id=clause.doc_id,
                    tender_id=clause.tender_id,
                    extraction_method="regex",
                    confidence=0.85,
                ))

    # Pass 2 — optional LLM fallback
    if enable_llm_fallback:
        from src.extract._llm_definitions import llm_extract_definitions

        candidates_for_llm = [
            c for c in clauses
            if (not llm_only_in_definitions_sections or _looks_like_definitions_section(c))
        ]
        # Skip clauses that already produced a regex hit to save tokens
        regex_hit_clauses = {d.defined_in_clause for d in found}
        candidates_for_llm = [c for c in candidates_for_llm if c.clause_id not in regex_hit_clauses]
        if candidates_for_llm:
            logger.info(
                "LLM fallback: %d candidate clauses (Definitions-section heuristic)",
                len(candidates_for_llm),
            )
            llm_extracted = llm_extract_definitions(candidates_for_llm)
            for d in llm_extracted:
                key = (d.canonical_form.lower(), d.doc_id)
                if key in seen_canonical:
                    continue
                seen_canonical.add(key)
                found.append(d)

    return found


# ─── I/O ────────────────────────────────────────────────────────────────────
def write_definitions(out_path: Path, definitions: list[Definition]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in definitions:
            f.write(d.model_dump_json() + "\n")


def read_definitions(in_path: Path) -> list[Definition]:
    in_path = Path(in_path)
    if not in_path.exists():
        return []
    out: list[Definition] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Definition.model_validate_json(line))
    return out
