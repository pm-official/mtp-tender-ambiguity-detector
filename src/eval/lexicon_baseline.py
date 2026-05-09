"""Rule-based vagueness-marker baseline (Type F detector).

A simple regex/lexicon detector that flags any clause containing one of a
fixed list of vagueness markers. This is the obvious baseline the BCT
pipeline must beat — without this comparison, reviewers will rightly ask
"does your fancy LLM pipeline outperform grep?".

The lexicon is grouped by failure mode so the per-marker contribution is
inspectable. References:

  Lewis, P. (2010), "Plain English in contract drafting" — discusses
    "best efforts", "reasonable", "as appropriate" as canonical vague
    contractual language.
  Hoek, R. and Schepens, A. (2018) "Computational analysis of vague
    obligations in construction contracts" — uses a 38-term lexicon.

Usage from CLI:
    python -m src.cli baseline lexicon --tender-id SYN_001
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.schemas.clause import Clause

logger = logging.getLogger(__name__)


# ─── Lexicon ───────────────────────────────────────────────────────────────
# Markers grouped by failure mode. Phrasing is deliberately permissive
# (whole-word match, case-insensitive). Each marker is a regex.
LEXICON: dict[str, list[str]] = {
    "subjectivity": [
        r"\breasonable\b",
        r"\breasonably\b",
        r"\badequate\b",
        r"\badequately\b",
        r"\bsuitable\b",
        r"\bappropriate\b",
        r"\bappropriately\b",
        r"\bsufficient\b",
        r"\bsufficiently\b",
        r"\bsatisfactory\b",
        r"\bproper\b",
        r"\bproperly\b",
    ],
    "effort_pledges": [
        r"\bbest\s+effort(?:s)?\b",
        r"\breasonable\s+effort(?:s)?\b",
        r"\ball\s+reasonable\s+steps?\b",
        r"\bgood\s+faith\b",
        r"\bdue\s+diligence\b",
    ],
    "delegated_authority": [
        r"\bas\s+(?:directed|required|requested|approved)\b",
        r"\bsubject\s+to\s+(?:the\s+)?approval\b",
        r"\bin\s+(?:the\s+)?opinion\s+of\b",
        r"\bat\s+(?:the\s+)?discretion\s+of\b",
        r"\bacceptable\s+to\s+(?:the\s+)?\w+",   # "acceptable to the Authority"
    ],
    "approximation": [
        r"\bapproximate(?:ly)?\b",
        r"\bnearly\b",
        r"\baround\b",
        r"\bcirca\b",
        r"\b(?:approximately\s+)?\d+\s*(?:%|percent)\s+(?:or\s+(?:more|less)|\+/-|±)\b",
    ],
    "feasibility_dodge": [
        r"\bwherever\s+feasible\b",
        r"\bif\s+feasible\b",
        r"\bas\s+far\s+as\s+practicable\b",
        r"\bto\s+the\s+extent\s+practicable\b",
        r"\bas\s+(?:may\s+be\s+)?required\b",
    ],
    "underspecified_quantifier": [
        r"\bsome\b",
        r"\bany\s+(?:relevant|appropriate)\b",
        r"\bvarious\b",
        r"\bcertain\b",
        r"\bother\s+(?:relevant\s+)?(?:standards?|codes?|materials?)\b",
    ],
}


def _compile() -> dict[str, list[re.Pattern]]:
    return {
        category: [re.compile(p, re.IGNORECASE) for p in patterns]
        for category, patterns in LEXICON.items()
    }


_COMPILED = _compile()


def detect_markers(text: str) -> list[tuple[str, str]]:
    """Return [(category, matched_text), ...] for every marker in `text`.

    Multiple matches are returned (no dedupe) so the marker count can drive
    severity heuristics if needed."""
    out: list[tuple[str, str]] = []
    for category, patterns in _COMPILED.items():
        for p in patterns:
            for m in p.finditer(text):
                out.append((category, m.group(0)))
    return out


def is_flagged(text: str, *, min_markers: int = 1) -> bool:
    """Whether the lexicon detector would flag `text` as vague."""
    return len(detect_markers(text)) >= min_markers


# ─── Pipeline-style record ─────────────────────────────────────────────────
def baseline_flag_record(clause: Clause) -> dict:
    """Produce a flag record in the same shape as candidate_flags.jsonl rows
    so downstream tooling (metrics, reports) can consume both interchangeably.
    Only the fields needed for evaluation are populated; LLM-only fields
    (BCT output, severity, etc.) are left empty/None.
    """
    markers = detect_markers(clause.text)
    flagged = bool(markers)
    categories = sorted({c for c, _ in markers})
    return {
        "flag_id": f"{clause.doc_id}::lex::{clause.clause_id}",
        "clause_id": clause.clause_id,
        "doc_id": clause.doc_id,
        "tender_id": clause.tender_id,
        "page": clause.page,
        "clause_text": clause.text,
        "section_path": clause.section_path,
        "clause_number": clause.clause_number,
        "flagged": flagged,
        "flag_reason": "lexicon:" + "|".join(categories) if flagged else "",
        "marker_count": len(markers),
        "marker_categories": categories,
        "matched_markers": [m for _, m in markers],
        # The fields below are required by candidate_flags.jsonl consumers;
        # populated minimally so reports don't break.
        "bct_output": {"commitments": []},
    }


def run_baseline(
    tender_id: str,
    corpus_root: Optional[Path] = None,
    min_markers: int = 1,
) -> dict:
    """Run the lexicon baseline on a parsed tender's clauses.

    Returns a summary dict; writes the per-clause flag records to
    `corpus/<tender_id>/parsed/baseline_lexicon.jsonl`.
    """
    corpus_root = corpus_root or Path("corpus")
    parsed_dir = corpus_root / tender_id / "parsed"
    clauses_path = parsed_dir / "clauses.jsonl"
    if not clauses_path.exists():
        raise FileNotFoundError(
            f"clauses.jsonl not found at {clauses_path} — parse the tender first"
        )

    out_path = parsed_dir / "baseline_lexicon.jsonl"
    n_total = 0
    n_flagged = 0
    category_counts: dict[str, int] = {}

    with clauses_path.open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            clause = Clause(**json.loads(line))
            n_total += 1
            rec = baseline_flag_record(clause)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["flagged"] and rec["marker_count"] >= min_markers:
                n_flagged += 1
                for cat in rec["marker_categories"]:
                    category_counts[cat] = category_counts.get(cat, 0) + 1

    summary = {
        "tender_id": tender_id,
        "method": "lexicon_baseline",
        "clauses_total": n_total,
        "clauses_flagged": n_flagged,
        "flag_rate": round(n_flagged / max(n_total, 1), 4),
        "category_counts": category_counts,
        "min_markers": min_markers,
        "lexicon_size": sum(len(v) for v in LEXICON.values()),
        "out_path": str(out_path),
    }
    summary_path = parsed_dir / "baseline_lexicon_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Baseline lexicon: %s", summary)
    return summary


__all__ = [
    "LEXICON",
    "detect_markers",
    "is_flagged",
    "baseline_flag_record",
    "run_baseline",
]
