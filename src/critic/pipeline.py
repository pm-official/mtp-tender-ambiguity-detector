"""End-to-end Stage 2 (apparent-vs-real) for one tender.

Reads:
  corpus/<tender_id>/parsed/candidate_flags.jsonl    (Stage 1 output)
  corpus/<tender_id>/parsed/clauses.jsonl
  corpus/<tender_id>/parsed/definitions.jsonl       (optional)
  corpus/<tender_id>/parsed/quantities.jsonl        (optional)
  + ChromaDB collection at .cache/chroma/<tender_id>

Writes:
  corpus/<tender_id>/parsed/stage2_verdicts.jsonl
  corpus/<tender_id>/parsed/stage2_summary.json
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
from src.critic.stage2 import run_stage2_critic
from src.extract.definitions import read_definitions
from src.extract.quantities import read_quantities
from src.llm.gemini_client import LLMCallStats, get_default_client
from src.retrieve.hybrid import retrieve_for_flag
from src.schemas.bct import CandidateFlag
from src.schemas.clause import Clause
from src.schemas.stage2 import Stage2Verdict

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
        raise FileNotFoundError(
            f"{p} not found. Run `score run --tender-id <id>` first."
        )
    out: list[CandidateFlag] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(CandidateFlag.model_validate_json(line))
    return out


def run_stage2(
    tender_dir: Path,
    *,
    flagged_only: bool = True,
    max_flags: Optional[int] = None,
    top_k_fused: int = 8,
    model: Optional[str] = None,
) -> dict:
    """Run Stage 2 critic on all flagged clauses in a tender."""
    tender_dir = Path(tender_dir)
    parsed_dir = tender_dir / "parsed"

    clauses = _load_clauses(tender_dir)
    flags = _load_flags(tender_dir)
    definitions = read_definitions(parsed_dir / "definitions.jsonl")
    quantities = read_quantities(parsed_dir / "quantities.jsonl")
    tender_id = clauses[0].tender_id if clauses else tender_dir.name

    target_flags = [f for f in flags if f.flagged] if flagged_only else flags
    if max_flags is not None:
        target_flags = target_flags[:max_flags]

    # ── Open run manifest (Feature A) ─────────────────────────────────────
    client = get_default_client()
    manifest = start_run(
        tender_id=tender_id,
        stage="stage_2_critic",
        models={"critic": model or client.default_pro,
                "embedding": client.default_embedding},
        parameters={
            "flagged_only": flagged_only,
            "max_flags": max_flags,
            "top_k_fused": top_k_fused,
        },
    )
    record_input(manifest, parsed_dir / "candidate_flags.jsonl")
    record_input(manifest, parsed_dir / "clauses.jsonl")
    record_input(manifest, parsed_dir / "definitions.jsonl")
    record_input(manifest, parsed_dir / "quantities.jsonl")

    logger.info("Stage 2: running critic on %d flag(s)...", len(target_flags))

    verdicts: list[Stage2Verdict] = []
    total_cost_inr = 0.0
    failures = 0

    for i, flag in enumerate(target_flags, 1):
        t0 = time.monotonic()
        try:
            hits = retrieve_for_flag(
                flag,
                tender_id=tender_id,
                clauses=clauses,
                definitions=definitions,
                quantities=quantities,
                top_k_fused=top_k_fused,
            )
            verdict, stats = run_stage2_critic(flag, hits, model=model)
            verdict.seconds_elapsed = time.monotonic() - t0
            verdicts.append(verdict)
            total_cost_inr += stats.cost_inr
            record_llm_stats(manifest, stats)
        except Exception as e:
            logger.warning("Stage 2 failed for flag %s: %s", flag.flag_id, e)
            failures += 1
            verdicts.append(Stage2Verdict(
                flag_id=flag.flag_id,
                clause_id=flag.clause_id,
                verdict="REAL",   # conservative default on failure
                rationale=f"Stage 2 critic failed: {e}",
                confidence=0.0,
                error=str(e),
                seconds_elapsed=time.monotonic() - t0,
            ))

        if i % 5 == 0 or i == len(target_flags):
            logger.info(
                "Stage 2 progress: %d/%d  REAL/APPARENT/WEAK so far: %d/%d/%d  cost INR %.2f",
                i, len(target_flags),
                sum(1 for v in verdicts if v.verdict == "REAL"),
                sum(1 for v in verdicts if v.verdict == "APPARENT"),
                sum(1 for v in verdicts if v.verdict == "WEAK"),
                total_cost_inr,
            )

    out_path = parsed_dir / "stage2_verdicts.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(v.model_dump_json() + "\n")

    summary = {
        "tender_id": tender_id,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "candidate_flags_total": len(flags),
        "candidate_flags_flagged": sum(1 for f in flags if f.flagged),
        "stage2_calls": len(verdicts),
        "stage2_failures": failures,
        "verdict_real": sum(1 for v in verdicts if v.verdict == "REAL"),
        "verdict_apparent": sum(1 for v in verdicts if v.verdict == "APPARENT"),
        "verdict_weak": sum(1 for v in verdicts if v.verdict == "WEAK"),
        "total_cost_inr": round(total_cost_inr, 4),
        "out_path": str(out_path),
    }

    with (parsed_dir / "stage2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Close run manifest ────────────────────────────────────────────────
    record_output(manifest, out_path)
    record_output(manifest, parsed_dir / "stage2_summary.json")
    manifest.summary = {k: v for k, v in summary.items() if k not in ("out_path", "ran_at", "tender_id")}
    finish_run(manifest)
    write_manifest(tender_dir, manifest)
    return summary


def load_verdicts(tender_dir: Path) -> list[Stage2Verdict]:
    p = Path(tender_dir) / "parsed" / "stage2_verdicts.jsonl"
    if not p.exists():
        return []
    out: list[Stage2Verdict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Stage2Verdict.model_validate_json(line))
    return out
