"""CLI sub-commands for Stage 4 (IS-code-grounded rewrite)."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from src.rewrite.pipeline import load_rewrites, run_stage4


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
        print(f"[rewrite run] {folder} does not exist.")
        return 1

    print(f"[rewrite run] Stage 4 IS-code rewrite on {args.tender_id}...")
    summary = run_stage4(
        folder,
        max_flags=args.max,
        is_code_top_k=args.is_code_top_k,
        max_attempts=args.max_attempts,
        intent_threshold=args.intent_threshold,
        model=args.model,
        only_severity=args.only_severity,
    )

    print("\n[rewrite run] Done.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    rewrites = load_rewrites(folder)
    if not rewrites:
        print(f"[rewrite inspect] No stage4_rewrites.jsonl. Run `rewrite run` first.")
        return 1

    if args.status:
        rewrites = [r for r in rewrites if r.status == args.status.upper()]
    if args.accepted_only:
        rewrites = [r for r in rewrites if r.status == "ACCEPTED"]
    if not rewrites:
        print(f"[rewrite inspect] No matching rewrites after filters.")
        return 0

    n = min(args.n, len(rewrites))
    rng = random.Random(args.seed)
    sample = rng.sample(rewrites, n) if n <= len(rewrites) else rewrites

    print(f"\n[rewrite inspect] {n} of {len(rewrites)} rewrites from {args.tender_id}:\n")
    for i, r in enumerate(sample, 1):
        marker = {"ACCEPTED": "[ACCEPTED]",
                  "REJECTED": "[REJECTED]",
                  "PASSED_GUARDRAILS_FAILED": "[GUARDRAIL FAIL]",
                  "RETRY_LIMIT_REACHED": "[RETRY LIMIT]",
                  "ERROR": "[ERROR]"}.get(r.status, "[?]")

        print(f"  [{i:>2d}] {marker}  attempts={r.attempts}  cost=INR {r.cost_inr:.2f}  cosine={r.intent_cosine:.3f}")
        print(f"        flag:    {r.flag_id}")
        print(f"        ORIGINAL: {(r.original_text or '').replace(chr(10), ' ').strip()[:200]}{'...' if len(r.original_text)>200 else ''}")
        if r.rewrite_text:
            print(f"        REWRITE:  {r.rewrite_text.replace(chr(10), ' ').strip()[:280]}{'...' if len(r.rewrite_text)>280 else ''}")
        if r.is_code_citations:
            print(f"        CITATIONS:")
            for c in r.is_code_citations:
                mark = "OK " if c.resolved else "-- "
                v = c.version or "any"
                s = f"§{c.section}" if c.section else "(no section)"
                print(f"          {mark} {c.code} {v} {s}")
                if not c.resolved:
                    print(f"               reason: {c.rejection_reason}")
        if r.rewrite_explanation:
            print(f"        explanation: {r.rewrite_explanation[:200]}")
        if r.rejection_reason:
            print(f"        reject_reason: {r.rejection_reason[:240]}")
        print()

    accepted = sum(1 for r in rewrites if r.status == "ACCEPTED")
    print(f"  Summary: ACCEPTED={accepted}  out of {len(rewrites)}")
    return 0


def add_subparser(parent: argparse._SubParsersAction) -> None:
    rewrite = parent.add_parser("rewrite", help="Stage 4 — IS-code-grounded rewrite.")
    rewrite.add_argument("--corpus-root", type=Path, default=None)
    sub = rewrite.add_subparsers(dest="rewrite_command", required=True)

    p = sub.add_parser("run", help="Run Stage 4 on every CONFIRMED flag.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("--max", type=int, default=None,
                    help="Cap number of CONFIRMED flags rewritten.")
    p.add_argument("--is-code-top-k", type=int, default=6)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--intent-threshold", type=float, default=0.6)
    p.add_argument("--only-severity", choices=["high", "medium", "low"], default=None,
                    help="Only rewrite flags at this severity tier.")
    p.add_argument("--model", type=str, default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("inspect", help="Show N random rewrites.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--status", choices=["ACCEPTED", "REJECTED", "RETRY_LIMIT_REACHED",
                                          "PASSED_GUARDRAILS_FAILED", "ERROR"], default=None)
    p.add_argument("--accepted-only", action="store_true")
    p.set_defaults(func=cmd_inspect)
