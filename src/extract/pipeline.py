"""Stage-0 part 2 — extract structures from already-parsed clauses.jsonl.

Runs after src.parse.pipeline. Produces:

  corpus/<tender_id>/parsed/definitions.jsonl
  corpus/<tender_id>/parsed/quantities.jsonl
  corpus/<tender_id>/parsed/references.jsonl
  corpus/<tender_id>/parsed/reference_graph.json
  + ChromaDB collection at .cache/chroma/<tender_id>/

Each step is independent and skippable so we can iterate cheaply.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.extract.definitions import extract_definitions, write_definitions
from src.extract.quantities import (
    classify_quantity_referents,
    extract_quantities,
    write_quantities,
)
from src.extract.references import (
    build_reference_graph,
    extract_references,
    write_reference_graph,
    write_references,
)
from src.extract.vector_index import build_vector_index
from src.schemas.clause import Clause

logger = logging.getLogger(__name__)


def _load_clauses(tender_dir: Path) -> list[Clause]:
    p = tender_dir / "parsed" / "clauses.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `parse run --tender-id <id>` first."
        )
    out: list[Clause] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Clause.model_validate_json(line))
    return out


def extract_all(
    tender_dir: Path,
    *,
    do_index: bool = True,
    do_definitions: bool = True,
    do_quantities: bool = True,
    do_references: bool = True,
    classify_referents_with_llm: bool = False,
    llm_definitions_fallback: bool = False,
) -> dict:
    """Run every Session-4 extractor on a tender. Returns a summary dict."""
    tender_dir = Path(tender_dir)
    parsed_dir = tender_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    clauses = _load_clauses(tender_dir)
    tender_id = clauses[0].tender_id if clauses else tender_dir.name

    summary: dict = {"tender_id": tender_id, "clauses_loaded": len(clauses)}

    # ── 1. Definitions ────────────────────────────────────────────────────
    if do_definitions:
        defs = extract_definitions(
            clauses,
            enable_llm_fallback=llm_definitions_fallback,
        )
        write_definitions(parsed_dir / "definitions.jsonl", defs)
        summary["definitions_extracted"] = len(defs)
        logger.info("Definitions extracted: %d", len(defs))

    # ── 2. Quantities (regex first, optional LLM referent classification) ─
    if do_quantities:
        quants = extract_quantities(clauses)
        if classify_referents_with_llm and quants:
            logger.info("Classifying %d quantity referents via LLM...", len(quants))
            quants = classify_quantity_referents(quants)
        write_quantities(parsed_dir / "quantities.jsonl", quants)
        summary["quantities_extracted"] = len(quants)
        if classify_referents_with_llm:
            classified = sum(1 for q in quants if q.referent_method == "llm")
            summary["quantities_referent_classified"] = classified

    # ── 3. References + graph ─────────────────────────────────────────────
    if do_references:
        refs = extract_references(clauses)
        graph, refs = build_reference_graph(clauses, refs)
        write_references(parsed_dir / "references.jsonl", refs)
        write_reference_graph(parsed_dir / "reference_graph.json", graph)
        resolved = sum(1 for r in refs if r.resolved)
        summary["references_extracted"] = len(refs)
        summary["references_resolved_internal"] = resolved
        summary["graph_nodes"] = graph.number_of_nodes()
        summary["graph_edges"] = graph.number_of_edges()

    # ── 4. Vector index (live Gemini calls; cached aggressively) ──────────
    if do_index:
        result = build_vector_index(clauses, tender_id=tender_id)
        summary["index_collection"] = result.collection_name
        summary["index_total_clauses"] = result.total_in_collection
        summary["index_newly_embedded"] = result.newly_embedded
        summary["index_skipped"] = result.skipped_already_indexed
        summary["index_cost_usd"] = round(result.cost_usd, 6)
        summary["index_cost_inr"] = round(result.cost_inr, 4)

    # Persist a single summary file
    with (parsed_dir / "extract_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
