"""End-to-end Stage 3 (critique pass + severity scoring) for one tender.

Reads:
  corpus/<tender_id>/parsed/candidate_flags.jsonl
  corpus/<tender_id>/parsed/stage2_verdicts.jsonl
  corpus/<tender_id>/parsed/clauses.jsonl
  corpus/<tender_id>/parsed/definitions.jsonl
  corpus/<tender_id>/parsed/quantities.jsonl

Writes:
  corpus/<tender_id>/parsed/stage3_confirmed.jsonl
  corpus/<tender_id>/parsed/stage3_summary.json

For each clause that came out of Stage 2:
  - APPARENT or WEAK verdicts pass through Stage 3 unchanged (recorded but
    not critiqued or severity-scored — they are not REAL).
  - REAL verdicts run through:
      1. Critique pass (LLM call). Output: CONFIRMED / REJECTED / WEAK.
      2. If CONFIRMED, severity scoring (LLM call). Output: 3 × 1-5 scores
         + composite + tier label.
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
from src.critic.stage3 import run_critique_pass, run_severity_scoring
from src.extract.definitions import read_definitions
from src.extract.quantities import read_quantities
from src.llm.gemini_client import get_default_client
from src.retrieve.hybrid import retrieve_for_flag
from src.schemas.bct import CandidateFlag
from src.schemas.clause import Clause
from src.schemas.stage2 import Stage2Verdict
from src.schemas.stage3 import ConfirmedFlag, severity_tier

logger = logging.getLogger(__name__)


def _load_clauses(tender_dir: Path) -> list[Clause]:
    p = tender_dir / "parsed" / "clauses.jsonl"
    out: list[Clause] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Clause.model_validate_json(line))
    return out


def _load_flags(tender_dir: Path) -> list[CandidateFlag]:
    p = tender_dir / "parsed" / "candidate_flags.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `score run` first.")
    out: list[CandidateFlag] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(CandidateFlag.model_validate_json(line))
    return out


def _load_verdicts(tender_dir: Path) -> list[Stage2Verdict]:
    p = tender_dir / "parsed" / "stage2_verdicts.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `critique run` first.")
    out: list[Stage2Verdict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Stage2Verdict.model_validate_json(line))
    return out


def run_stage3(
    tender_dir: Path,
    *,
    skip_severity: bool = False,
    max_real_flags: Optional[int] = None,
    top_k_fused: int = 8,
    critique_model: Optional[str] = None,
    severity_model: Optional[str] = None,
) -> dict:
    """Run Stage 3 (critique pass + severity scoring) on a tender.

    Parameters
    ----------
    skip_severity : if True, only run the critique pass (skip severity scoring).
    max_real_flags : cap number of REAL flags processed (for cost-bounded dev runs).
    """
    tender_dir = Path(tender_dir)
    parsed_dir = tender_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    clauses = _load_clauses(tender_dir)
    flags = _load_flags(tender_dir)
    verdicts = _load_verdicts(tender_dir)
    definitions = read_definitions(parsed_dir / "definitions.jsonl")
    quantities = read_quantities(parsed_dir / "quantities.jsonl")
    tender_id = clauses[0].tender_id if clauses else tender_dir.name

    # Index for joining
    flag_by_id = {f.flag_id: f for f in flags}
    verdict_by_flag = {v.flag_id: v for v in verdicts}
    real_flag_ids = [v.flag_id for v in verdicts if v.verdict == "REAL" and not v.error]

    # Apply max-real cap
    if max_real_flags is not None:
        real_flag_ids = real_flag_ids[:max_real_flags]

    # ── Run manifest ──────────────────────────────────────────────────────
    client = get_default_client()
    manifest = start_run(
        tender_id=tender_id,
        stage="stage_3_critique_severity",
        models={
            "critique": critique_model or client.default_pro,
            "severity": severity_model or client.default_pro,
        },
        parameters={
            "skip_severity": skip_severity,
            "max_real_flags": max_real_flags,
            "top_k_fused": top_k_fused,
        },
    )
    record_input(manifest, parsed_dir / "candidate_flags.jsonl")
    record_input(manifest, parsed_dir / "stage2_verdicts.jsonl")

    logger.info(
        "Stage 3: running critique on %d Stage-2 REAL flags...",
        len(real_flag_ids),
    )

    confirmed_records: list[ConfirmedFlag] = []
    critique_failures = 0
    severity_failures = 0
    total_cost_inr = 0.0

    # Carry every Stage-2 verdict forward as a ConfirmedFlag record so downstream
    # stages have a complete picture (REAL+critique outcome, plus APPARENT/WEAK
    # passthrough records for completeness).
    started = time.monotonic()

    for v in verdicts:
        flag = flag_by_id.get(v.flag_id)
        if flag is None:
            logger.warning("Verdict references missing flag %s", v.flag_id)
            continue

        cf = ConfirmedFlag(
            flag_id=flag.flag_id,
            clause_id=flag.clause_id,
            doc_id=flag.doc_id,
            tender_id=flag.tender_id,
            page=flag.page,
            clause_text=flag.clause_text,
            section_path=flag.section_path,
            clause_number=flag.clause_number,
            stage2_verdict=v.verdict,
            stage2_rationale=v.rationale,
            resolving_clause_id=v.resolving_clause_id,
        )

        # Stage 3 only critiques REAL verdicts (the others are pass-through)
        if v.verdict != "REAL" or v.error or v.flag_id not in real_flag_ids:
            confirmed_records.append(cf)
            continue

        # Re-run hybrid retrieval (cheap; ChromaDB is local) to give critic same context as Stage 2
        try:
            hits = retrieve_for_flag(
                flag,
                tender_id=tender_id,
                clauses=clauses,
                definitions=definitions,
                quantities=quantities,
                top_k_fused=top_k_fused,
            )
        except Exception as e:
            logger.warning("Retrieval failed for %s: %s; skipping critique", flag.flag_id, e)
            cf.error = f"retrieval failed: {e}"
            confirmed_records.append(cf)
            continue

        # ── Critique pass ────────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            critique, stats = run_critique_pass(flag, v, hits, model=critique_model)
            cf.critique_verdict = critique.verdict
            cf.critique_refutation = critique.refutation
            if critique.resolving_clause_id:
                cf.resolving_clause_id = critique.resolving_clause_id
            record_llm_stats(manifest, stats)
            total_cost_inr += stats.cost_inr
        except Exception as e:
            logger.warning("Critique failed for %s: %s", flag.flag_id, e)
            cf.error = f"critique failed: {e}"
            critique_failures += 1
            confirmed_records.append(cf)
            continue

        # ── Severity scoring (only if CONFIRMED + not skipped) ───────────
        if not skip_severity and cf.critique_verdict == "CONFIRMED":
            try:
                severity, sev_stats = run_severity_scoring(flag, model=severity_model)
                cf.severity = severity
                cf.composite_severity = severity.composite_severity
                cf.severity_tier = severity_tier(severity.composite_severity)
                record_llm_stats(manifest, sev_stats)
                total_cost_inr += sev_stats.cost_inr
            except Exception as e:
                logger.warning("Severity scoring failed for %s: %s", flag.flag_id, e)
                severity_failures += 1
                cf.error = f"severity scoring failed: {e}"

        cf.seconds_elapsed = time.monotonic() - t0
        cf.cost_inr = sum(s.get("cost_inr", 0) or 0 for s in [])  # placeholder; tracked in manifest
        confirmed_records.append(cf)

        if (len([r for r in confirmed_records if r.critique_verdict]) % 5) == 0:
            done = sum(1 for r in confirmed_records if r.critique_verdict)
            confirmed_count = sum(1 for r in confirmed_records if r.critique_verdict == "CONFIRMED")
            rejected_count = sum(1 for r in confirmed_records if r.critique_verdict == "REJECTED")
            weak_count = sum(1 for r in confirmed_records if r.critique_verdict == "WEAK")
            logger.info(
                "Stage 3 progress: %d critiqued (CONF/REJ/WEAK = %d/%d/%d) cost INR %.2f",
                done, confirmed_count, rejected_count, weak_count, total_cost_inr,
            )

    # ── Persist ──────────────────────────────────────────────────────────
    out_path = parsed_dir / "stage3_confirmed.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for cf in confirmed_records:
            f.write(cf.model_dump_json() + "\n")

    # ── Summary ──────────────────────────────────────────────────────────
    critiqued = [r for r in confirmed_records if r.critique_verdict]
    confirmed = [r for r in confirmed_records if r.critique_verdict == "CONFIRMED"]
    rejected = [r for r in confirmed_records if r.critique_verdict == "REJECTED"]
    weak = [r for r in confirmed_records if r.critique_verdict == "WEAK"]

    severity_distribution = {"high": 0, "medium": 0, "low": 0}
    for cf in confirmed:
        if cf.severity_tier in severity_distribution:
            severity_distribution[cf.severity_tier] += 1

    rejection_rate = round(len(rejected) / max(len(critiqued), 1), 3)

    summary = {
        "tender_id": tender_id,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(confirmed_records),
        "stage2_real_input": len(real_flag_ids),
        "critique_calls": len(critiqued),
        "critique_failures": critique_failures,
        "critique_confirmed": len(confirmed),
        "critique_rejected": len(rejected),
        "critique_weak": len(weak),
        "critique_rejection_rate": rejection_rate,
        "severity_calls": sum(1 for r in confirmed_records if r.severity is not None),
        "severity_failures": severity_failures,
        "severity_distribution": severity_distribution,
        "total_cost_inr": round(total_cost_inr, 4),
        "out_path": str(out_path),
    }

    with (parsed_dir / "stage3_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Manifest finish ───────────────────────────────────────────────────
    record_output(manifest, out_path)
    record_output(manifest, parsed_dir / "stage3_summary.json")
    manifest.summary = {k: v for k, v in summary.items() if k not in ("out_path", "ran_at", "tender_id")}
    finish_run(manifest)
    write_manifest(tender_dir, manifest)

    return summary


def load_confirmed(tender_dir: Path) -> list[ConfirmedFlag]:
    p = Path(tender_dir) / "parsed" / "stage3_confirmed.jsonl"
    if not p.exists():
        return []
    out: list[ConfirmedFlag] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ConfirmedFlag.model_validate_json(line))
    return out
