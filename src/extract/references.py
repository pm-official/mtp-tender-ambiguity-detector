"""Cross-reference graph builder.

Detects every "look elsewhere" pointer in tender clauses and builds a
directed graph of (from_clause -> target). Targets can be:

  * Internal clauses ("Clause 4.6", "Sub-Clause 1.5", "Section 5")
  * Annexes / Appendices / Schedules / Drawings
  * External standards ("IS 456:2000", "MoRTH 5.0", "IRC SP-13")
  * Generic external pointers ("as per the relevant code")

The graph is persisted as a node-link JSON via NetworkX, and as a flat
references.jsonl alongside it for stages that just want a list.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

from src.schemas.clause import Clause
from src.schemas.structures import Reference

logger = logging.getLogger(__name__)


# ─── Patterns ───────────────────────────────────────────────────────────────
# Each pattern captures group "label" = the human-readable target string.

_REFERENCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Internal clause refs
    (re.compile(r"\b(?:Sub-?\s*Clause|Clause)\s+(?P<label>\d+(?:\.\d+)+[A-Za-z]?)\b", re.IGNORECASE), "clause"),
    (re.compile(r"\bSection\s+(?P<label>\d+(?:\.\d+)*)\b", re.IGNORECASE), "section"),
    (re.compile(r"\bAnnex(?:ure)?\s*[-:]?\s*(?P<label>[IVXLCDM]+|[A-Z]|\d+)\b", re.IGNORECASE), "annex"),
    (re.compile(r"\bAppendix\s*[-:]?\s*(?P<label>[A-Z]|[IVXLCDM]+|\d+)\b", re.IGNORECASE), "appendix"),
    (re.compile(r"\bSchedule\s*[-:]?\s*(?P<label>[A-Z]|[IVXLCDM]+|\d+)\b", re.IGNORECASE), "schedule"),
    (re.compile(r"\bDrawing(?:\s+No\.?)?\s*[-:]?\s*(?P<label>[A-Z0-9\-\.]+)\b", re.IGNORECASE), "drawing"),
    # Standards
    (re.compile(r"\bIS\s*(?P<label>\d+(?:\.\d+)?(?:\s*:\s*\d{4})?(?:\s*Part\s*\d+)?)", re.IGNORECASE), "standard"),
    (re.compile(r"\bIRC\s*[-:]?\s*(?P<label>SP[-\s]*\d+|\d+)", re.IGNORECASE), "standard"),
    (re.compile(r"\bMoRTH(?:\s+\d+\.\d+)?", re.IGNORECASE), "standard"),
    (re.compile(r"\bCPWD\s+(?P<label>\d{4}|GS|Specifications?|Works?\s+Manual)?", re.IGNORECASE), "standard"),
    (re.compile(r"\bNBC\s*(?P<label>\d{4})?", re.IGNORECASE), "standard"),
]


# ─── Public API ─────────────────────────────────────────────────────────────
def extract_references(clauses: Iterable[Clause]) -> list[Reference]:
    """Find every reference in every clause. Does not resolve targets — that
    is done by `build_reference_graph` once we have the full clause list."""
    refs: list[Reference] = []
    for clause in clauses:
        for pat, kind in _REFERENCE_PATTERNS:
            for m in pat.finditer(clause.text):
                raw = m.group(0).strip()
                label_group = m.groupdict().get("label", "")
                label_text = (label_group or raw).strip()
                if not label_text:
                    label_text = raw
                target_label = _format_target_label(kind, label_text, raw)
                refs.append(Reference(
                    from_clause=clause.clause_id,
                    target_kind=kind,
                    target_label=target_label,
                    raw_text=raw,
                    confidence=0.9,
                ))
    return refs


def _format_target_label(kind: str, label_text: str, raw: str) -> str:
    label_text = re.sub(r"\s+", " ", label_text).strip()
    if kind == "clause":
        return f"Clause {label_text}"
    if kind == "section":
        return f"Section {label_text}"
    if kind == "annex":
        return f"Annex {label_text}"
    if kind == "appendix":
        return f"Appendix {label_text}"
    if kind == "schedule":
        return f"Schedule {label_text}"
    if kind == "drawing":
        return f"Drawing {label_text}"
    if kind == "standard":
        return raw or label_text
    return label_text


# ─── Graph construction ─────────────────────────────────────────────────────
def build_reference_graph(
    clauses: Iterable[Clause],
    references: list[Reference],
):
    """Build a NetworkX DiGraph linking clauses to their resolved targets.

    Internal-clause refs ("Clause 4.6") are resolved to actual clause_ids
    where possible by matching on (doc_id, clause_number). External and
    unresolvable refs become "external" nodes.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx not installed") from e

    g = nx.DiGraph()
    clause_list = list(clauses)

    # Index clauses by (doc_id, clause_number) for resolution
    by_number: dict[tuple[str, str], str] = {}
    for c in clause_list:
        if c.clause_number:
            by_number[(c.doc_id, c.clause_number)] = c.clause_id
        # Add the clause as a node
        g.add_node(
            c.clause_id,
            kind="clause",
            doc_id=c.doc_id,
            clause_number=c.clause_number or "",
            section=" > ".join(c.section_path),
        )

    for ref in references:
        from_id = ref.from_clause
        # Try to resolve internal references by clause_number within the same doc
        target_id: Optional[str] = None
        if ref.target_kind == "clause":
            num = ref.target_label.replace("Clause ", "").strip()
            from_clause = next((c for c in clause_list if c.clause_id == from_id), None)
            if from_clause:
                target_id = by_number.get((from_clause.doc_id, num))

        if target_id and target_id in g:
            g.add_edge(from_id, target_id, kind=ref.target_kind, label=ref.target_label)
            ref.target_id = target_id
            ref.resolved = True
        else:
            # Add target as an external node and link
            target_node = f"external::{ref.target_kind}::{ref.target_label}"
            g.add_node(
                target_node,
                kind=ref.target_kind,
                external=True,
                label=ref.target_label,
            )
            g.add_edge(from_id, target_node, kind=ref.target_kind, label=ref.target_label)

    return g, references


# ─── I/O ────────────────────────────────────────────────────────────────────
def write_references(out_path: Path, refs: list[Reference]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in refs:
            f.write(r.model_dump_json() + "\n")


def write_reference_graph(out_path: Path, g) -> None:
    """Persist NetworkX graph as JSON node-link format."""
    import networkx as nx

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(g, edges="edges")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_references(in_path: Path) -> list[Reference]:
    in_path = Path(in_path)
    if not in_path.exists():
        return []
    out: list[Reference] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Reference.model_validate_json(line))
    return out
