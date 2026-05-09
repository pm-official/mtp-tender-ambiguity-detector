"""CLI sub-commands for the end-to-end pipeline orchestrator.

Wired into `python -m src.cli pipeline ...` via src/cli.py.

Sub-commands:
  run     — run all stages 0 → 2 on a tender (and 3, 4 once built)
  status  — print which artefacts exist for a tender + last-run summary
  history — show run history from runs/_index.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.pipeline import pipeline_status, run_pipeline


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
        print(f"[pipeline run] {folder} does not exist.")
        return 1

    print(f"[pipeline run] End-to-end pipeline on {args.tender_id}...")
    summary = run_pipeline(
        folder,
        do_parse=not args.skip_parse,
        do_extract=not args.skip_extract,
        do_extract_index=not args.skip_extract_index,
        do_extract_definitions=not args.skip_extract_definitions,
        do_extract_quantities=not args.skip_extract_quantities,
        do_extract_references=not args.skip_extract_references,
        do_score=not args.skip_score,
        do_critique=not args.skip_critique,
        do_confirm=not args.skip_confirm,
        do_rewrite=not args.skip_rewrite,
        rewrite_max=args.rewrite_max,
        rewrite_only_severity=args.rewrite_only_severity,
        rewrite_model=args.rewrite_model,
        classify_referents_with_llm=args.classify_referents,
        llm_definitions_fallback=args.llm_definitions,
        score_max=args.score_max,
        score_skip_filter=args.score_skip_filter,
        critique_max=args.critique_max,
        critique_top_k=args.critique_top_k,
        confirm_max=args.confirm_max,
        confirm_skip_severity=args.confirm_skip_severity,
        score_model=args.score_model,
        critique_model=args.critique_model,
        confirm_critique_model=args.confirm_critique_model,
        confirm_severity_model=args.confirm_severity_model,
    )

    print("\n[pipeline run] Done.\n")
    print(json.dumps(summary, indent=2)[:3000])
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    if not folder.exists():
        print(f"[pipeline status] {folder} does not exist.")
        return 1

    status = pipeline_status(folder)
    print(f"\n[pipeline status] {args.tender_id}\n")
    for label, info in status.items():
        if isinstance(info, dict) and "present" in info:
            mark = "[OK]  " if info["present"] else "[--]  "
            extra = ""
            if info.get("present"):
                if "lines" in info:
                    extra = f"{info['lines']:>6d} lines, {info['size_bytes']:>10d} bytes"
                else:
                    extra = f"          {info['size_bytes']:>10d} bytes"
            print(f"  {mark}{label:<35} {extra}")
    if "past_runs" in status:
        print(f"\n  past pipeline runs: {status['past_runs']}")
        if status.get("latest_run"):
            r = status["latest_run"]
            print(f"  latest run: {r.get('stage')} at {r.get('started_at')} ({r.get('seconds_elapsed', 0):.1f}s, INR {r.get('cost_inr', 0):.4f})")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    from src.audit.run_manifest import list_runs

    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    if not folder.exists():
        print(f"[pipeline history] {folder} does not exist.")
        return 1

    runs = list_runs(folder)
    if not runs:
        print(f"[pipeline history] No runs recorded for {args.tender_id}.")
        return 0

    print(f"\n[pipeline history] {len(runs)} run(s) for {args.tender_id}:\n")
    print(f"  {'started':<22} {'stage':<22} {'sec':>6} {'cost INR':>10} {'cache hit':>10}  ok")
    print(f"  {'-' * 22} {'-' * 22} {'-' * 6} {'-' * 10} {'-' * 10}  --")
    for r in runs[-args.limit:]:
        ok = "OK" if r.get("success") else "ERR"
        cache_rate = r.get("cache_hit_rate", 0)
        print(
            f"  {r.get('started_at', '?'):<22} {r.get('stage', '?'):<22} "
            f"{r.get('seconds_elapsed', 0):>6.1f} {r.get('cost_inr', 0):>10.4f} {cache_rate:>10.1%}  {ok}"
        )
    return 0


def add_subparser(parent: argparse._SubParsersAction) -> None:
    pipeline = parent.add_parser(
        "pipeline",
        help="End-to-end pipeline orchestrator.",
    )
    pipeline.add_argument(
        "--corpus-root", type=Path, default=None,
        help="Override the default corpus/ directory.",
    )
    sub = pipeline.add_subparsers(dest="pipeline_command", required=True)

    p = sub.add_parser("run", help="Run all stages on a tender.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("--skip-parse", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-extract-index", action="store_true")
    p.add_argument("--skip-extract-definitions", action="store_true")
    p.add_argument("--skip-extract-quantities", action="store_true")
    p.add_argument("--skip-extract-references", action="store_true")
    p.add_argument("--skip-score", action="store_true")
    p.add_argument("--skip-critique", action="store_true")
    p.add_argument("--skip-confirm", action="store_true")
    p.add_argument("--skip-rewrite", action="store_true")
    p.add_argument("--rewrite-max", type=int, default=None)
    p.add_argument("--rewrite-only-severity", choices=["high", "medium", "low"], default=None)
    p.add_argument("--rewrite-model", type=str, default=None)
    p.add_argument("--classify-referents", action="store_true")
    p.add_argument("--llm-definitions", action="store_true")
    p.add_argument("--score-max", type=int, default=None)
    p.add_argument("--score-skip-filter", action="store_true")
    p.add_argument("--critique-max", type=int, default=None)
    p.add_argument("--critique-top-k", type=int, default=8)
    p.add_argument("--confirm-max", type=int, default=None,
                    help="Cap REAL flags Stage 3 critiques (cost-bounded dev runs).")
    p.add_argument("--confirm-skip-severity", action="store_true")
    p.add_argument("--score-model", type=str, default=None)
    p.add_argument("--critique-model", type=str, default=None)
    p.add_argument("--confirm-critique-model", type=str, default=None)
    p.add_argument("--confirm-severity-model", type=str, default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="Print which artefacts exist for a tender.")
    p.add_argument("--tender-id", required=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("history", help="Show pipeline run history for a tender.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_history)
