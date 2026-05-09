"""End-to-end Stage 1 (BCT) for one tender.

Reads corpus/<tender_id>/parsed/clauses.jsonl, runs the obligation filter,
runs BCT on every obligation clause, applies the decision rule, and writes:

  corpus/<tender_id>/parsed/candidate_flags.jsonl    one CandidateFlag per line
  corpus/<tender_id>/parsed/stage1_summary.json      aggregate stats

Note: non-obligation clauses are recorded with `is_obligation=False`,
`flagged=False`, and an empty bct_output. Downstream stages should filter
on `flagged=True` to get the "candidate ambiguities" set.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.audit.run_manifest import (
    finish_run,
    record_input,
    record_llm_stats,
    record_output,
    start_run,
    write_manifest,
)
from src.llm.gemini_client import get_default_client
from src.schemas.bct import BCTOutput, CandidateFlag, evaluate_flag
from src.schemas.clause import Clause
from src.score.obligation_filter import filter_obligations
from src.score.stage1_bct import run_bct_on_clause

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


def _short_clause_id(cid: str) -> str:
    """Extract a short slug from a clause_id for use in flag_id."""
    if "::" in cid:
        return cid.split("::")[-1]
    return cid[:12]


def run_stage1(
    tender_dir: Path,
    *,
    skip_obligation_filter: bool = False,
    skip_table_clauses: bool = True,
    min_clause_chars: int = 40,
    max_clauses: Optional[int] = None,
    model: Optional[str] = None,
) -> dict:
    """Run Stage 1 BCT on every obligation clause in a tender.

    Parameters
    ----------
    skip_obligation_filter : bool
        If True, run BCT on every clause (more expensive, useful for debugging).
    skip_table_clauses : bool
        If True, skip clauses whose text came from a table row. Tables are
        useful for numerical-inconsistency detection (a future stage) but
        are noisy for BCT.
    min_clause_chars : int
        Skip clauses shorter than this (typically table-cell artefacts).
    max_clauses : int | None
        Cap the number of clauses BCT is run on (useful for cost-bounded
        development runs).
    """
    tender_dir = Path(tender_dir)
    parsed_dir = tender_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    clauses = _load_clauses(tender_dir)
    tender_id = clauses[0].tender_id if clauses else tender_dir.name

    # ── Open run manifest (Feature A) ─────────────────────────────────────
    client = get_default_client()
    manifest = start_run(
        tender_id=tender_id,
        stage="stage_1_bct",
        models={"obligation_filter": client.default_flash, "bct": model or client.default_flash},
        parameters={
            "skip_obligation_filter": skip_obligation_filter,
            "skip_table_clauses": skip_table_clauses,
            "min_clause_chars": min_clause_chars,
            "max_clauses": max_clauses,
        },
    )
    record_input(manifest, parsed_dir / "clauses.jsonl")

    # ── Pre-filter: drop table rows and tiny clauses by default ───────────
    eligible: list[Clause] = []
    skipped_table = 0
    skipped_tiny = 0
    for c in clauses:
        if skip_table_clauses and c.table_origin:
            skipped_table += 1
            continue
        if c.char_count < min_clause_chars:
            skipped_tiny += 1
            continue
        eligible.append(c)
    if max_clauses is not None:
        eligible = eligible[:max_clauses]

    logger.info(
        "Stage 1: %d clauses loaded → %d eligible after pre-filter "
        "(skipped %d tables, %d tiny)",
        len(clauses), len(eligible), skipped_table, skipped_tiny,
    )

    # ── Step 1: obligation filter ─────────────────────────────────────────
    if skip_obligation_filter:
        obligation_map = {
            c.clause_id: type("X", (), {"is_obligation": True, "rationale": "filter skipped"})()
            for c in eligible
        }
    else:
        logger.info("Step 1: obligation filter on %d clauses...", len(eligible))
        obligation_map = filter_obligations(eligible, model=model)

    obligation_clauses = [c for c in eligible if obligation_map[c.clause_id].is_obligation]
    skipped_non_obligation = len(eligible) - len(obligation_clauses)
    logger.info(
        "Step 1 done: %d obligations, %d non-obligations skipped",
        len(obligation_clauses), skipped_non_obligation,
    )

    # ── Step 2: BCT on every obligation clause ───────────────────────────
    flags: list[CandidateFlag] = []
    total_cost_inr = 0.0
    bct_failures = 0

    started = time.monotonic()
    for i, clause in enumerate(obligation_clauses, 1):
        t0 = time.monotonic()
        try:
            bct_out, stats = run_bct_on_clause(clause, model=model)
        except Exception as e:
            logger.warning("BCT call failed for %s: %s", clause.clause_id, e)
            bct_failures += 1
            continue

        record_llm_stats(manifest, stats)

        flagged, reason = evaluate_flag(bct_out)
        flag = CandidateFlag(
            flag_id=f"{clause.doc_id}::flag::{_short_clause_id(clause.clause_id)}",
            clause_id=clause.clause_id,
            doc_id=clause.doc_id,
            tender_id=clause.tender_id,
            page=clause.page,
            clause_text=clause.text,
            section_path=clause.section_path,
            clause_number=clause.clause_number,
            bct_output=bct_out,
            flagged=flagged,
            flag_reason=reason,
            is_obligation=True,
            cost_inr=stats.cost_inr,
            seconds_elapsed=time.monotonic() - t0,
        )
        flags.append(flag)
        total_cost_inr += stats.cost_inr

        if i % 25 == 0 or i == len(obligation_clauses):
            elapsed = time.monotonic() - started
            logger.info(
                "BCT progress: %d/%d  flags so far: %d  total cost: INR %.2f  elapsed: %.1fs",
                i, len(obligation_clauses),
                sum(1 for f in flags if f.flagged),
                total_cost_inr, elapsed,
            )

    # ── Also record non-obligation clauses (audit completeness) ──────────
    for c in eligible:
        if c.clause_id in {f.clause_id for f in flags}:
            continue
        check = obligation_map[c.clause_id]
        flags.append(CandidateFlag(
            flag_id=f"{c.doc_id}::flag::{_short_clause_id(c.clause_id)}",
            clause_id=c.clause_id,
            doc_id=c.doc_id,
            tender_id=c.tender_id,
            page=c.page,
            clause_text=c.text,
            section_path=c.section_path,
            clause_number=c.clause_number,
            bct_output=BCTOutput(commitments=[]),
            flagged=False,
            flag_reason="not an obligation",
            is_obligation=False,
            skipped_reason=getattr(check, "rationale", None) or "obligation filter said no",
        ))

    # ── Persist ──────────────────────────────────────────────────────────
    out_path = parsed_dir / "candidate_flags.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for fl in flags:
            f.write(fl.model_dump_json() + "\n")

    summary = {
        "tender_id": tender_id,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "clauses_loaded": len(clauses),
        "skipped_tables": skipped_table,
        "skipped_tiny": skipped_tiny,
        "eligible": len(eligible),
        "obligation_filter_skipped": skipped_non_obligation,
        "bct_calls": len([f for f in flags if f.is_obligation]),
        "bct_failures": bct_failures,
        "flagged": sum(1 for f in flags if f.flagged),
        "not_flagged": sum(1 for f in flags if f.is_obligation and not f.flagged),
        "non_obligation_records": skipped_non_obligation,
        "total_cost_inr": round(total_cost_inr, 4),
        "out_path": str(out_path),
    }

    with (parsed_dir / "stage1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Close run manifest ────────────────────────────────────────────────
    record_output(manifest, out_path)
    record_output(manifest, parsed_dir / "stage1_summary.json")
    manifest.summary = {k: v for k, v in summary.items() if k not in ("out_path", "ran_at", "tender_id")}
    finish_run(manifest)
    write_manifest(tender_dir, manifest)
    return summary


def load_flags(tender_dir: Path) -> list[CandidateFlag]:
    p = Path(tender_dir) / "parsed" / "candidate_flags.jsonl"
    if not p.exists():
        return []
    out: list[CandidateFlag] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(CandidateFlag.model_validate_json(line))
    return out
