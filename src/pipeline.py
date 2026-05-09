"""End-to-end pipeline orchestrator.

Runs every available stage on a tender, in order, with sensible defaults.
Each stage is opt-out via flags so users can re-run individual stages cheaply.

The orchestrator records a single top-level RunManifest in addition to the
per-stage manifests written by each stage. Together these form the
provenance record needed for paper-grade reproducibility.

Usage:
    python -m src.cli pipeline run --tender-id SYN_001
    python -m src.cli pipeline run --tender-id JK_001 --skip-extract-index
    python -m src.cli pipeline status --tender-id JK_001
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def run_pipeline(
    tender_dir: Path,
    *,
    do_parse: bool = True,
    do_extract: bool = True,
    do_extract_index: bool = True,
    do_extract_definitions: bool = True,
    do_extract_quantities: bool = True,
    do_extract_references: bool = True,
    do_score: bool = True,
    do_critique: bool = True,
    do_confirm: bool = True,
    do_rewrite: bool = True,
    classify_referents_with_llm: bool = False,
    llm_definitions_fallback: bool = False,
    score_max: Optional[int] = None,
    score_skip_filter: bool = False,
    critique_max: Optional[int] = None,
    critique_top_k: int = 8,
    confirm_max: Optional[int] = None,
    confirm_skip_severity: bool = False,
    rewrite_max: Optional[int] = None,
    rewrite_only_severity: Optional[str] = None,
    score_model: Optional[str] = None,
    critique_model: Optional[str] = None,
    confirm_critique_model: Optional[str] = None,
    confirm_severity_model: Optional[str] = None,
    rewrite_model: Optional[str] = None,
) -> dict:
    """Run all stages 0 → 2 (and 3, 4 once built) on a tender.

    Stages 5 (silver labels) and 8/9 (critique severity, rewrite) will be
    plumbed in here as those sessions complete.
    """
    from src.audit.run_manifest import (
        finish_run,
        record_input,
        record_output,
        start_run,
        write_manifest,
    )
    from src.critic.pipeline import run_stage2
    from src.critic.stage3_pipeline import run_stage3
    from src.extract.pipeline import extract_all
    from src.parse.pipeline import parse_tender
    from src.rewrite.pipeline import run_stage4
    from src.score.pipeline import run_stage1

    tender_dir = Path(tender_dir)
    tender_id = tender_dir.name

    summary: dict[str, Any] = {
        "tender_id": tender_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages": {},
    }

    # Top-level manifest
    top_manifest = start_run(
        tender_id=tender_id,
        stage="pipeline_full",
        parameters={
            "do_parse": do_parse,
            "do_extract": do_extract,
            "do_extract_index": do_extract_index,
            "do_score": do_score,
            "do_critique": do_critique,
            "score_max": score_max,
            "critique_max": critique_max,
            "critique_top_k": critique_top_k,
        },
    )

    # ── Stage 0a: parse ─────────────────────────────────────────────────────
    if do_parse:
        logger.info("─── Stage 0a: parsing ───")
        parse_summary = parse_tender(tender_dir)
        summary["stages"]["parse"] = parse_summary
    else:
        logger.info("Skipping parse (do_parse=False)")

    # ── Stage 0b: extract structures ────────────────────────────────────────
    if do_extract:
        logger.info("─── Stage 0b: extracting structures ───")
        extract_summary = extract_all(
            tender_dir,
            do_index=do_extract_index,
            do_definitions=do_extract_definitions,
            do_quantities=do_extract_quantities,
            do_references=do_extract_references,
            classify_referents_with_llm=classify_referents_with_llm,
            llm_definitions_fallback=llm_definitions_fallback,
        )
        summary["stages"]["extract"] = extract_summary
    else:
        logger.info("Skipping extract (do_extract=False)")

    # ── Stage 1: BCT ────────────────────────────────────────────────────────
    if do_score:
        logger.info("─── Stage 1: BCT ───")
        score_summary = run_stage1(
            tender_dir,
            skip_obligation_filter=score_skip_filter,
            max_clauses=score_max,
            model=score_model,
        )
        summary["stages"]["score"] = score_summary
    else:
        logger.info("Skipping score (do_score=False)")

    # ── Stage 2: apparent-vs-real critic ────────────────────────────────────
    if do_critique:
        logger.info("─── Stage 2: apparent-vs-real critic ───")
        try:
            critique_summary = run_stage2(
                tender_dir,
                max_flags=critique_max,
                top_k_fused=critique_top_k,
                model=critique_model,
            )
            summary["stages"]["critique"] = critique_summary
        except FileNotFoundError as e:
            logger.warning("Skipping critique: %s", e)
            summary["stages"]["critique"] = {"skipped": str(e)}
    else:
        logger.info("Skipping critique (do_critique=False)")

    # ── Stage 3: critique pass + severity scoring ────────────────────────
    if do_confirm:
        logger.info("─── Stage 3: critique pass + severity scoring ───")
        try:
            confirm_summary = run_stage3(
                tender_dir,
                skip_severity=confirm_skip_severity,
                max_real_flags=confirm_max,
                top_k_fused=critique_top_k,
                critique_model=confirm_critique_model,
                severity_model=confirm_severity_model,
            )
            summary["stages"]["confirm"] = confirm_summary
        except FileNotFoundError as e:
            logger.warning("Skipping confirm: %s", e)
            summary["stages"]["confirm"] = {"skipped": str(e)}
    else:
        logger.info("Skipping confirm (do_confirm=False)")

    # ── Stage 4: IS-code-grounded rewrite ────────────────────────────────
    if do_rewrite:
        logger.info("─── Stage 4: IS-code-grounded rewrite ───")
        try:
            rewrite_summary = run_stage4(
                tender_dir,
                max_flags=rewrite_max,
                only_severity=rewrite_only_severity,
                model=rewrite_model,
            )
            summary["stages"]["rewrite"] = rewrite_summary
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("Skipping rewrite: %s", e)
            summary["stages"]["rewrite"] = {"skipped": str(e)}
    else:
        logger.info("Skipping rewrite (do_rewrite=False)")

    summary["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── Top-level manifest summary ──────────────────────────────────────────
    top_manifest.summary = _summarise(summary)
    finish_run(top_manifest)
    write_manifest(tender_dir, top_manifest)

    return summary


def _summarise(summary: dict) -> dict:
    """Compact summary of stage outputs for the top-level run-manifest record."""
    stages = summary.get("stages", {})
    return {
        "parse_clauses": stages.get("parse", {}).get("total_clauses"),
        "parse_table_clauses": stages.get("parse", {}).get("table_clauses"),
        "extract_definitions": stages.get("extract", {}).get("definitions_extracted"),
        "extract_quantities": stages.get("extract", {}).get("quantities_extracted"),
        "extract_references": stages.get("extract", {}).get("references_extracted"),
        "extract_index_total": stages.get("extract", {}).get("index_total_clauses"),
        "score_eligible": stages.get("score", {}).get("eligible"),
        "score_flagged": stages.get("score", {}).get("flagged"),
        "score_cost_inr": stages.get("score", {}).get("total_cost_inr"),
        "critique_real": stages.get("critique", {}).get("verdict_real"),
        "critique_apparent": stages.get("critique", {}).get("verdict_apparent"),
        "critique_weak": stages.get("critique", {}).get("verdict_weak"),
        "critique_cost_inr": stages.get("critique", {}).get("total_cost_inr"),
        "confirm_confirmed": stages.get("confirm", {}).get("critique_confirmed"),
        "confirm_rejected": stages.get("confirm", {}).get("critique_rejected"),
        "confirm_weak": stages.get("confirm", {}).get("critique_weak"),
        "confirm_severity_distribution": stages.get("confirm", {}).get("severity_distribution"),
        "confirm_cost_inr": stages.get("confirm", {}).get("total_cost_inr"),
        "rewrite_status_counts": stages.get("rewrite", {}).get("status_counts"),
        "rewrite_guardrails": stages.get("rewrite", {}).get("guardrails_passed"),
        "rewrite_cost_inr": stages.get("rewrite", {}).get("total_cost_inr"),
    }


# ─── Status / pipeline_status ───────────────────────────────────────────────
def pipeline_status(tender_dir: Path) -> dict:
    """Return what artefacts exist for the tender — a quick sanity check."""
    tender_dir = Path(tender_dir)
    parsed = tender_dir / "parsed"
    runs = tender_dir / "runs"

    out: dict[str, Any] = {"tender_id": tender_dir.name}

    artefacts = {
        "manifest.json":              tender_dir / "manifest.json",
        "parsed/clauses.jsonl":       parsed / "clauses.jsonl",
        "parsed/document_tree.json":  parsed / "document_tree.json",
        "parsed/definitions.jsonl":   parsed / "definitions.jsonl",
        "parsed/quantities.jsonl":    parsed / "quantities.jsonl",
        "parsed/references.jsonl":    parsed / "references.jsonl",
        "parsed/reference_graph.json":parsed / "reference_graph.json",
        "parsed/extract_summary.json":parsed / "extract_summary.json",
        "parsed/candidate_flags.jsonl":parsed / "candidate_flags.jsonl",
        "parsed/stage1_summary.json": parsed / "stage1_summary.json",
        "parsed/stage2_verdicts.jsonl":parsed / "stage2_verdicts.jsonl",
        "parsed/stage2_summary.json": parsed / "stage2_summary.json",
        "parsed/stage3_confirmed.jsonl":parsed / "stage3_confirmed.jsonl",
        "parsed/stage3_summary.json": parsed / "stage3_summary.json",
        "parsed/stage4_rewrites.jsonl":parsed / "stage4_rewrites.jsonl",
        "parsed/stage4_summary.json": parsed / "stage4_summary.json",
        "runs/_index.jsonl":          runs / "_index.jsonl",
    }

    for label, p in artefacts.items():
        if not p.exists():
            out[label] = {"present": False}
            continue
        info: dict[str, Any] = {"present": True, "size_bytes": p.stat().st_size}
        if p.suffix.lower() == ".jsonl":
            info["lines"] = sum(1 for _ in p.open("r", encoding="utf-8"))
        out[label] = info

    # Past runs
    from src.audit.run_manifest import list_runs
    runs_index = list_runs(tender_dir)
    out["past_runs"] = len(runs_index)
    if runs_index:
        out["latest_run"] = runs_index[-1]
    return out
