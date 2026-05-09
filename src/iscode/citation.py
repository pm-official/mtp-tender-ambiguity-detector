"""Citation parser + verifier (Stage 4 guardrail G1).

Parses IS-code-style citations from rewrite text and verifies each against
the local IS-code registry. A citation passes if the corpus contains a
chunk for (code_id_normalised, section) — either the exact section or any
sub-section starting with it.

We support a few citation styles common in Indian construction practice:

  IS 456:2000 Section 8.1
  IS 456:2000 §8.1
  IS 456 Section 8.1            (no version specified; we infer the latest)
  IS 1200 (Part 5):1982 §3
  IS 1200-Part5-1982 §3
  CPWD Specifications 2019 Vol 1 §3.2
  IS 456:2000 (Cl. 8.1)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.iscode.index import load_registry
from src.schemas.rewrite import ISCodeCitation

logger = logging.getLogger(__name__)


# ─── Citation regexes ──────────────────────────────────────────────────────
# IS-style: "IS 456:2000 §8.1" — primary number after the family.
_IS_CITATION_PATTERN = re.compile(
    r"""
    \b
    (?P<family>IS)
    [\s–—-]+
    (?P<num>\d+)
    (?:\s*\(?Part\s*(?P<part>\d+)\)?)?
    (?:\s*[:–—-]\s*(?P<year>\d{4}))?
    (?:
       \s*(?:Section|Sec\.?|Clause|Cl\.?|§|\(Cl\.?\s*)
       \s*(?P<section>\d+(?:\.\d+){0,5})\)?
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# CPWD-style: "CPWD Specifications:2019 §5.4". The family is followed by a
# Title (CapWords) rather than digits, since CPWD codes are named.
# The title excludes citation-marker words (Section/Clause/Sec/Cl/Vol) via
# a negative lookahead so "CPWD GCC Section 104" parses as title="GCC",
# section="104", not as title="GCC Section" with no section.
_CPWD_CITATION_PATTERN = re.compile(
    r"""
    \b
    (?P<family>CPWD)
    [\s–—-]+
    (?P<title>
       (?!(?:Section|Sec|Clause|Cl|Vol)\b)
       [A-Z][A-Za-z]+
       (?:\s+(?!(?:Section|Sec|Clause|Cl|Vol)\b)[A-Z][A-Za-z]+)*
    )
    (?:\s*[:–—-]\s*(?P<year>\d{4}))?
    (?:\s*Vol\.?\s*(?P<vol>\d+))?
    (?:
       \s*(?:Section|Sec\.?|Clause|Cl\.?|§|\(Cl\.?\s*)
       \s*(?P<section>\d+(?:\.\d+){0,5})\)?
    )?
    """,
    re.VERBOSE,
)


def parse_citations(text: str) -> list[ISCodeCitation]:
    """Find every IS / CPWD citation in `text`. Returns one ISCodeCitation
    per match, with raw fields filled in (resolved=False at this point)."""
    out: list[ISCodeCitation] = []

    for m in _IS_CITATION_PATTERN.finditer(text):
        num = m.group("num")
        part = m.group("part")
        year = m.group("year")
        section = m.group("section")
        code = f"IS {num} (Part {part})" if part else f"IS {num}"
        out.append(ISCodeCitation(
            code=code, version=year, section=section, raw_text=m.group(0),
        ))

    for m in _CPWD_CITATION_PATTERN.finditer(text):
        title = m.group("title")
        year = m.group("year")
        vol = m.group("vol")
        section = m.group("section")
        code = f"CPWD {title}"
        if vol:
            code = f"{code} Vol{vol}"
        out.append(ISCodeCitation(
            code=code, version=year, section=section, raw_text=m.group(0),
        ))

    return _dedupe(out)


def _dedupe(citations: list[ISCodeCitation]) -> list[ISCodeCitation]:
    seen: set[tuple] = set()
    out: list[ISCodeCitation] = []
    for c in citations:
        key = (c.code, c.version, c.section)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ─── Verification ──────────────────────────────────────────────────────────
def _normalise_code_id(code: str, version: Optional[str]) -> str:
    """Match the registry's code_id format: 'IS 456:2000', 'CPWD …:2019'."""
    code_clean = re.sub(r"\s+", " ", code.strip())
    if version:
        return f"{code_clean}:{version}"
    return code_clean


def verify_citation(c: ISCodeCitation, registry: Optional[dict] = None) -> ISCodeCitation:
    """Look up a citation in the registry. Mutates and returns the citation."""
    if registry is None:
        registry = load_registry()
    if not registry:
        c.resolved = False
        c.rejection_reason = "registry empty (run `iscode build` first)"
        return c

    if not c.section:
        # Citation has no section; treat as code-only — accept if any chunk
        # of that code exists in the registry.
        prefix = _normalise_code_id(c.code, c.version)
        for key in registry:
            if key.startswith(prefix + "::"):
                c.resolved = True
                c.resolved_chunk_id = registry[key]["chunk_id"]
                return c
        # No version specified — try matching code_id without the version
        if not c.version:
            for key in registry:
                if key.startswith(c.code + ":") or key.startswith(c.code + "::"):
                    c.resolved = True
                    c.resolved_chunk_id = registry[key]["chunk_id"]
                    return c
        c.rejection_reason = f"no chunks found for code '{c.code}' (version {c.version or 'any'})"
        return c

    # Citation has a section — try exact match first, then prefix match
    # (for sub-sections), and finally version-relaxed match.
    for code_attempt in [
        _normalise_code_id(c.code, c.version),
        c.code,   # in case version wasn't given but registry has versioned key
    ]:
        # Exact match
        key = f"{code_attempt}::{c.section}"
        if key in registry:
            c.resolved = True
            c.resolved_chunk_id = registry[key]["chunk_id"]
            return c
        # Prefix match (citation says §8 but registry has §8.1, §8.2)
        for k in registry:
            if k.startswith(f"{code_attempt}::{c.section}."):
                c.resolved = True
                c.resolved_chunk_id = registry[k]["chunk_id"]
                return c
        # Version-relaxed: any version of the code with the same section
        for k in registry:
            registry_code, registry_section = k.split("::", 1)
            if registry_code.startswith(code_attempt.split(":")[0]) \
                    and registry_section == c.section:
                c.resolved = True
                c.resolved_chunk_id = registry[k]["chunk_id"]
                c.rejection_reason = ""  # accepted via version-relaxed
                return c

    c.rejection_reason = (
        f"no chunk found for '{c.code} {c.version or ''}' §{c.section} "
        f"(checked exact, prefix, version-relaxed)"
    )
    return c


def verify_all_citations(citations: list[ISCodeCitation]) -> tuple[list[ISCodeCitation], int, int]:
    """Verify every citation. Returns (citations_with_resolved_set,
    resolved_count, unresolved_count)."""
    registry = load_registry()
    out: list[ISCodeCitation] = []
    resolved = 0
    unresolved = 0
    for c in citations:
        v = verify_citation(c, registry=registry)
        out.append(v)
        if v.resolved:
            resolved += 1
        else:
            unresolved += 1
    return out, resolved, unresolved
