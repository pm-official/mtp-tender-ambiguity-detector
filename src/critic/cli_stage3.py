"""CLI sub-commands for Stage 3 (critique pass + severity scoring)."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from src.critic.stage3_pipeline import load_confirmed, run_stage3


def _project_corpus_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent / "corpus"
    return Path.cwd() / "corpus"


def cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    if not folder.exists():
        print(f"[confirm run] {folder} does not exist.")
        return 1

    print(f"[confirm run] Stage 3 critique + severity on {args.tender_id}...")
    summary = run_stage3(
        folder,
        skip_severity=args.skip_severity,
        max_real_flags=args.max,
        top_k_fused=args.top_k,
        critique_model=args.critique_model,
        severity_model=args.severity_model,
    )

    print("\n[confirm run] Done.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    records = load_confirmed(folder)
    if not records:
        print(f"[confirm inspect] No stage3_confirmed.jsonl. Run `confirm run` first.")
        return 1

    if args.tier:
        records = [r for r in records if r.severity_tier == args.tier.lower()]
    if args.critique:
        records = [r for r in records if r.critique_verdict == args.critique.upper()]
    if args.confirmed_only:
        records = [r for r in records if r.critique_verdict == "CONFIRMED"]

    if not records:
        print(f"[confirm inspect] No matching records after filters.")
        return 0

    n = min(args.n, len(records))
    rng = random.Random(args.seed)
    sample = rng.sample(records, n) if n <= len(records) else records

    print(f"\n[confirm inspect] {n} of {len(records)} records from {args.tender_id}:\n")
    for i, r in enumerate(sample, 1):
        section = " > ".join(r.section_path or ["(none)"])
        clause_short = r.clause_text.replace("\n", " ").strip()
        if len(clause_short) > 200:
            clause_short = clause_short[:200] + " ..."

        s2 = f"S2={r.stage2_verdict}"
        if r.critique_verdict:
            s3 = f"S3={r.critique_verdict}"
        else:
            s3 = "S3=(passthrough)"
        sev = ""
        if r.severity:
            sev = f"  severity={r.composite_severity:>2}/10 [{r.severity_tier}]"

        print(f"  [{i:>2d}] {s2} -> {s3}{sev}")
        print(f"       clause:    {clause_short}")
        if r.critique_refutation:
            print(f"       refutation: {r.critique_refutation[:300]}")
        if r.severity:
            sv = r.severity
            print(f"       severity:   commercial={sv.commercial_exposure} dispute={sv.dispute_likelihood} reviewer_cost={sv.reviewer_cost}")
            print(f"       rationale:  {sv.rationale[:200]}")
        if r.resolving_clause_id:
            print(f"       resolved_by: {r.resolving_clause_id}")
        if r.error:
            print(f"       error:     {r.error[:200]}")
        print()

    # Summary footer
    confirmed = sum(1 for r in records if r.critique_verdict == "CONFIRMED")
    rejected = sum(1 for r in records if r.critique_verdict == "REJECTED")
    weak = sum(1 for r in records if r.critique_verdict == "WEAK")
    print(f"  Summary: CONFIRMED={confirmed}  REJECTED={rejected}  WEAK={weak}")
    return 0


def add_subparser(parent: argparse._SubParsersAction) -> None:
    confirm = parent.add_parser(
        "confirm",
        help="Stage 3 — critique pass + severity scoring.",
    )
    confirm.add_argument("--corpus-root", type=Path, default=None)
    sub = confirm.add_subparsers(dest="confirm_command", required=True)

    p = sub.add_parser("run", help="Run critique pass + severity scoring.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("--skip-severity", action="store_true",
                    help="Run only the critique pass; skip severity scoring.")
    p.add_argument("--max", type=int, default=None,
                    help="Cap number of REAL flags processed (cost-bounded dev runs).")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--critique-model", type=str, default=None)
    p.add_argument("--severity-model", type=str, default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("inspect", help="Show N random Stage-3 records.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tier", choices=["high", "medium", "low"], default=None)
    p.add_argument("--critique", choices=["CONFIRMED", "REJECTED", "WEAK", "confirmed", "rejected", "weak"], default=None)
    p.add_argument("--confirmed-only", action="store_true")
    p.set_defaults(func=cmd_inspect)
