"""CLI sub-commands for Stage 1 (BCT).

Wired into `python -m src.cli score ...` via src/cli.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from src.score.pipeline import load_flags, run_stage1

logger = logging.getLogger(__name__)


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
        print(f"[score run] {folder} does not exist.")
        return 1

    print(f"[score run] Stage 1 BCT on tender {args.tender_id}...")
    summary = run_stage1(
        folder,
        skip_obligation_filter=args.skip_filter,
        skip_table_clauses=not args.include_tables,
        max_clauses=args.max,
    )

    print("\n[score run] Done.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    flags = load_flags(folder)
    if not flags:
        print(f"[score inspect] No candidate_flags.jsonl found for {args.tender_id}. Run `score run` first.")
        return 1

    if args.flagged_only:
        flags = [f for f in flags if f.flagged]
    if args.obligations_only:
        flags = [f for f in flags if f.is_obligation]
    if not flags:
        print(f"[score inspect] No matching flags after filters.")
        return 0

    n = min(args.n, len(flags))
    rng = random.Random(args.seed)
    sample = rng.sample(flags, n) if n <= len(flags) else flags

    print(f"\n[score inspect] {n} of {len(flags)} flags from {args.tender_id}:\n")
    for i, f in enumerate(sample, 1):
        section = " > ".join(f.section_path or ["(none)"])
        marker = "[FLAG]" if f.flagged else ("[OK]" if f.is_obligation else "[skip]")
        print(f"  [{i:>2d}] {marker}  {f.doc_id}  page {f.page}  "
              f"clause={f.clause_number or '-':<10}  section={section[:40]}")
        text = f.clause_text.replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:200] + " ..."
        print(f"        clause:  {text}")
        if f.flagged:
            print(f"        reason:  {f.flag_reason}")
            for j, com in enumerate(f.bct_output.commitments, 1):
                print(f"        commitment {j}: {com.obligation[:90]}")
                print(f"           quantity: {com.quantity[:80]}")
                print(f"           method:   {com.method[:80]}")
                print(f"           standard: {com.standard[:80]}")
                if com.missing_info:
                    print(f"           missing_info ({len(com.missing_info)}):")
                    for q in com.missing_info[:3]:
                        print(f"             - {q[:120]}")
        elif f.is_obligation:
            print(f"        reason:  {f.flag_reason}")
        else:
            print(f"        skipped: {f.skipped_reason}")
        print()

    # Footer summary
    flagged_count = sum(1 for f in flags if f.flagged)
    oblig_count = sum(1 for f in flags if f.is_obligation)
    non_count = sum(1 for f in flags if not f.is_obligation)
    print(f"  Summary: flagged={flagged_count}  not-flagged-obligation={oblig_count - flagged_count}  non-obligation={non_count}")
    return 0


def add_subparser(parent: argparse._SubParsersAction) -> None:
    score = parent.add_parser(
        "score",
        help="Stage 1 — Bidder Commitment Test (BCT).",
    )
    score.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Override the default corpus/ directory.",
    )
    sub = score.add_subparsers(dest="score_command", required=True)

    p = sub.add_parser("run", help="Run BCT on every obligation clause in a tender.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("--max", type=int, default=None,
                    help="Cap number of obligation clauses (cost-bounded dev runs).")
    p.add_argument("--include-tables", action="store_true",
                    help="Don't skip table-origin clauses (default: skip).")
    p.add_argument("--skip-filter", action="store_true",
                    help="Skip the obligation filter (run BCT on every eligible clause).")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("inspect", help="Show N random candidate flags.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flagged-only", action="store_true",
                    help="Only show flagged clauses.")
    p.add_argument("--obligations-only", action="store_true",
                    help="Only show clauses that passed the obligation filter.")
    p.set_defaults(func=cmd_inspect)
