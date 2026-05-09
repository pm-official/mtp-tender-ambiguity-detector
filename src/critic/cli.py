"""CLI sub-commands for Stage 2 (apparent-vs-real)."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from src.critic.pipeline import load_verdicts, run_stage2
from src.score.pipeline import load_flags


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
        print(f"[critique run] {folder} does not exist.")
        return 1

    print(f"[critique run] Stage 2 apparent-vs-real on {args.tender_id}...")
    summary = run_stage2(
        folder,
        flagged_only=not args.all_flags,
        max_flags=args.max,
        top_k_fused=args.top_k,
        model=args.model,
    )
    print("\n[critique run] Done.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    verdicts = load_verdicts(folder)
    if not verdicts:
        print(f"[critique inspect] No stage2_verdicts.jsonl. Run `critique run` first.")
        return 1

    flags = {f.flag_id: f for f in load_flags(folder)}

    if args.verdict:
        verdicts = [v for v in verdicts if v.verdict == args.verdict.upper()]
        if not verdicts:
            print(f"[critique inspect] No verdicts of type {args.verdict.upper()}.")
            return 0

    n = min(args.n, len(verdicts))
    rng = random.Random(args.seed)
    sample = rng.sample(verdicts, n) if n <= len(verdicts) else verdicts

    print(f"\n[critique inspect] {n} of {len(verdicts)} verdicts from {args.tender_id}:\n")
    for i, v in enumerate(sample, 1):
        flag = flags.get(v.flag_id)
        clause_text = flag.clause_text if flag else "(flag not found)"
        if len(clause_text) > 200:
            clause_text = clause_text[:200] + " ..."
        clause_text = clause_text.replace("\n", " ").strip()

        marker = {"REAL": "[REAL]", "APPARENT": "[APPARENT]", "WEAK": "[WEAK]"}.get(v.verdict, "[?]")
        print(f"  [{i:>2d}] {marker:<10} confidence={v.confidence:.2f}")
        print(f"        flag:    {v.flag_id}")
        print(f"        clause:  {clause_text}")
        print(f"        verdict: {v.verdict}")
        if v.resolving_clause_id:
            print(f"        resolved by: {v.resolving_clause_id}")
        print(f"        rationale: {v.rationale[:300]}")
        if v.retrieved_chunk_ids:
            print(f"        retrieved {len(v.retrieved_chunk_ids)} chunks; routes used: ", end="")
            unique_routes = sorted({s for ss in v.retrieved_sources for s in ss})
            print(", ".join(unique_routes))
        print()

    print(f"  Summary: REAL={sum(1 for v in verdicts if v.verdict=='REAL')}  "
          f"APPARENT={sum(1 for v in verdicts if v.verdict=='APPARENT')}  "
          f"WEAK={sum(1 for v in verdicts if v.verdict=='WEAK')}")
    return 0


def add_subparser(parent: argparse._SubParsersAction) -> None:
    critique = parent.add_parser(
        "critique",
        help="Stage 2 — apparent-vs-real ambiguity adjudication.",
    )
    critique.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
    )
    sub = critique.add_subparsers(dest="critique_command", required=True)

    p = sub.add_parser("run", help="Run Stage 2 critic on every flagged clause.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("--max", type=int, default=None,
                    help="Cap number of flags critiqued (cost-bounded dev runs).")
    p.add_argument("--all-flags", action="store_true",
                    help="Critique non-flagged clauses too (audit-completeness mode).")
    p.add_argument("--top-k", type=int, default=8,
                    help="Top-K fused retrieval hits passed to the critic (default 8).")
    p.add_argument("--model", type=str, default=None,
                    help="Override the default Gemini model (e.g. gemini-2.5-pro).")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("inspect", help="Show N random verdicts.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verdict", choices=["REAL", "APPARENT", "WEAK", "real", "apparent", "weak"],
                    default=None, help="Filter to one verdict.")
    p.set_defaults(func=cmd_inspect)
