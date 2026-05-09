"""End-to-end Stage 4 (IS-code-grounded rewrite) for one tender.

Reads:
  corpus/<tender_id>/parsed/stage3_confirmed.jsonl
  corpus/<tender_id>/parsed/candidate_flags.jsonl   (BCT input recovery)
  + IS-code corpus at .cache/chroma/iscode_corpus/

Writes:
  corpus/<tender_id>/parsed/stage4_rewrites.jsonl
  corpus/<tender_id>/parsed/stage4_summary.json
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
from src.critic.stage3_pipeline import load_confirmed
from src.iscode.index import index_summary
from src.llm.gemini_client import get_default_client
from src.rewrite.stage4 import (
    DEFAULT_INTENT_COSINE_THRESHOLD,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TOP_K_ISCODE,
    rewrite_one,
)
from src.schemas.bct import CandidateFlag
from src.schemas.rewrite import Rewrite

logger = logging.getLogger(__name__)


def _load_flags(tender_dir: Path) -> dict[str, CandidateFlag]:
    p = tender_dir / "parsed" / "candidate_flags.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `score run` first.")
    out: dict[str, CandidateFlag] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                fl = CandidateFlag.model_validate_json(line)
                out[fl.flag_id] = fl
    return out


def run_stage4(
    tender_dir: Path,
    *,
    max_flags: Optional[int] = None,
    is_code_top_k: int = DEFAULT_TOP_K_ISCODE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    intent_threshold: float = DEFAULT_INTENT_COSINE_THRESHOLD,
    model: Optional[str] = None,
    only_severity: Optional[str] = None,    # "high" | "medium" | "low"
) -> dict:
    """Generate IS-code-grounded rewrites for every CONFIRMED flag.

    Parameters
    ----------
    only_severity : if set, only rewrite flags at that severity tier
                    (useful for cost-bounded runs that only target HIGH).
    """
    tender_dir = Path(tender_dir)
    parsed_dir = tender_dir / "parsed"

    summary_iscode = index_summary()
    if not summary_iscode.get("exists"):
        raise RuntimeError(
            "IS-code index not found. Run `python -m src.cli iscode build` first."
        )

    confirmed = load_confirmed(tender_dir)
    flag_by_id = _load_flags(tender_dir)

    candidates = [
        cf for cf in confirmed
        if cf.critique_verdict == "CONFIRMED" and cf.flag_id in flag_by_id
    ]
    if only_severity:
        candidates = [cf for cf in candidates if cf.severity_tier == only_severity]
    if max_flags is not None:
        candidates = candidates[:max_flags]

    # ── Manifest ──────────────────────────────────────────────────────────
    client = get_default_client()
    manifest = start_run(
        tender_id=candidates[0].tender_id if candidates else tender_dir.name,
        stage="stage_4_rewrite",
        models={
            "rewrite": model or client.default_pro,
            "embedding": client.default_embedding,
        },
        parameters={
            "max_flags": max_flags,
            "is_code_top_k": is_code_top_k,
            "max_attempts": max_attempts,
            "intent_threshold": intent_threshold,
            "only_severity": only_severity,
        },
    )
    record_input(manifest, parsed_dir / "candidate_flags.jsonl")
    record_input(manifest, parsed_dir / "stage3_confirmed.jsonl")

    logger.info(
        "Stage 4: generating rewrites for %d CONFIRMED flag(s)...", len(candidates),
    )

    rewrites: list[Rewrite] = []
    total_cost_inr = 0.0
    started = time.monotonic()

    for i, cf in enumerate(candidates, 1):
        flag = flag_by_id.get(cf.flag_id)
        if flag is None:
            continue
        try:
            rewrite, stats_list = rewrite_one(
                cf, flag,
                is_code_top_k=is_code_top_k,
                max_attempts=max_attempts,
                intent_threshold=intent_threshold,
                model=model,
            )
            for s in stats_list:
                record_llm_stats(manifest, s)
                total_cost_inr += s.cost_inr
            rewrites.append(rewrite)
            logger.info(
                "Stage 4 [%d/%d] %s -> %s (attempts=%d, INR %.2f, intent=%.3f)",
                i, len(candidates), flag.flag_id, rewrite.status,
                rewrite.attempts, rewrite.cost_inr, rewrite.intent_cosine,
            )
        except Exception as e:
            logger.warning("Stage 4 errored on %s: %s", cf.flag_id, e)
            rewrites.append(Rewrite(
                flag_id=cf.flag_id,
                clause_id=cf.clause_id,
                tender_id=cf.tender_id,
                doc_id=cf.doc_id,
                original_text=cf.clause_text,
                status="ERROR",
                error=str(e),
                rejection_reason=f"orchestrator error: {e}",
            ))

    # ── Persist ───────────────────────────────────────────────────────────
    out_path = parsed_dir / "stage4_rewrites.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rewrites:
            f.write(r.model_dump_json() + "\n")

    by_status = {"ACCEPTED": 0, "REJECTED": 0, "PASSED_GUARDRAILS_FAILED": 0,
                 "RETRY_LIMIT_REACHED": 0, "ERROR": 0}
    g1_pass = sum(1 for r in rewrites if r.citation_integrity_passed is True)
    g2_pass = sum(1 for r in rewrites if r.reflag_check_passed is True)
    g3_pass = sum(1 for r in rewrites if r.intent_check_passed is True)
    for r in rewrites:
        if r.status in by_status:
            by_status[r.status] += 1

    summary = {
        "tender_id": candidates[0].tender_id if candidates else tender_dir.name,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "candidates_total": len(candidates),
        "rewrites_attempted": len(rewrites),
        "status_counts": by_status,
        "guardrails_passed": {
            "citation_integrity": g1_pass,
            "reflag_check": g2_pass,
            "intent_preservation": g3_pass,
        },
        "total_cost_inr": round(total_cost_inr, 4),
        "out_path": str(out_path),
    }

    with (parsed_dir / "stage4_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    record_output(manifest, out_path)
    record_output(manifest, parsed_dir / "stage4_summary.json")
    manifest.summary = {k: v for k, v in summary.items() if k not in ("out_path", "ran_at", "tender_id")}
    finish_run(manifest)
    write_manifest(tender_dir, manifest)

    return summary


def load_rewrites(tender_dir: Path) -> list[Rewrite]:
    p = Path(tender_dir) / "parsed" / "stage4_rewrites.jsonl"
    if not p.exists():
        return []
    out: list[Rewrite] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Rewrite.model_validate_json(line))
    return out
