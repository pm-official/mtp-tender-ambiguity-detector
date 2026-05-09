"""CLI subcommands for the eval module.

  python -m src.cli eval lexicon  --tender-id SYN_001
  python -m src.cli eval metrics  --tender-id SYN_001 [--ground-truth path.txt]
  python -m src.cli eval compare  --tender-id SYN_001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── eval lexicon ──────────────────────────────────────────────────────────
def _cmd_lexicon(args: argparse.Namespace) -> int:
    from src.eval.lexicon_baseline import run_baseline

    summary = run_baseline(
        tender_id=args.tender_id,
        corpus_root=Path(args.corpus_root) if args.corpus_root else None,
        min_markers=args.min_markers,
    )
    print("[eval lexicon] Done.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


# ─── eval metrics ──────────────────────────────────────────────────────────
def _load_ground_truth(spec: str, tender_id: str) -> set[str]:
    """Resolve the --ground-truth argument:

      - "synthetic" → use the built-in hand-labels for SYN_001 etc.
      - <path>      → file with one clause_id per line, or a JSON list
    """
    if spec == "synthetic":
        from src.eval.synthetic_groundtruth import get_ground_truth
        gt = get_ground_truth(tender_id)
        if gt is None:
            raise SystemExit(
                f"No synthetic ground truth registered for tender_id "
                f"{tender_id!r}. See src/eval/synthetic_groundtruth.py."
            )
        return set(gt)
    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"Ground-truth file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return set(json.loads(text))
    return {line.strip() for line in text.splitlines() if line.strip()}


def _cmd_metrics(args: argparse.Namespace) -> int:
    from src.eval.metrics import eval_tender

    gt: set[str] | None = None
    if args.ground_truth:
        gt = _load_ground_truth(args.ground_truth, args.tender_id)

    # If we have a synthetic ground truth, restrict the universe to its
    # labelled clauses. Otherwise the metrics use the union of pipeline-
    # observed clauses as the universe (which inflates TN counts with
    # document-headers and tables).
    universe_override: set[str] | None = None
    if args.ground_truth == "synthetic":
        from src.eval.synthetic_groundtruth import SYN_001_UNIVERSE
        if args.tender_id == "SYN_001":
            universe_override = set(SYN_001_UNIVERSE)

    out = eval_tender(
        tender_id=args.tender_id,
        corpus_root=Path(args.corpus_root) if args.corpus_root else None,
        ground_truth=gt,
        universe_override=universe_override,
    )
    print(json.dumps(out, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[eval metrics] wrote {args.out}", file=sys.stderr)
    return 0


# ─── eval ablation ─────────────────────────────────────────────────────────
def _cmd_ablation(args: argparse.Namespace) -> int:
    """Run the four-rung ablation: no-filter / lexicon / BCT / +Stage2 / +Stage3."""
    from src.eval.ablation import format_ablation_table, run_ablation
    from src.eval.synthetic_groundtruth import (
        SYN_001_UNIVERSE,
        SYN_001_VAGUE_CLAUSE_IDS,
        category_counts,
    )

    if args.tender_id != "SYN_001":
        raise SystemExit(
            "Ablation currently requires synthetic ground truth, which is "
            f"only registered for SYN_001 (got {args.tender_id!r}). See "
            "src/eval/synthetic_groundtruth.py to add another."
        )

    gt = set(SYN_001_VAGUE_CLAUSE_IDS)
    universe = set(SYN_001_UNIVERSE)

    rows = run_ablation(
        tender_id=args.tender_id,
        corpus_root=Path(args.corpus_root) if args.corpus_root else None,
        ground_truth=gt,
        universe=universe,
    )

    print(f"\n  {args.tender_id} ablation -- universe={len(universe)}, "
          f"vague={len(gt)}\n")
    print(category_counts())
    print()
    print(format_ablation_table(rows))

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n[eval ablation] wrote {args.out}", file=sys.stderr)
    return 0


# ─── eval compare ──────────────────────────────────────────────────────────
def _cmd_compare(args: argparse.Namespace) -> int:
    """Print a side-by-side table: BCT vs lexicon, per category."""
    from src.eval.metrics import eval_tender

    out = eval_tender(
        tender_id=args.tender_id,
        corpus_root=Path(args.corpus_root) if args.corpus_root else None,
    )
    counts = out["counts"]
    agree = out["agreement"]["bct_vs_lexicon"]
    print(f"\n  {args.tender_id} -- universe={out['universe_size']} clauses\n")
    print("  Method                Flagged")
    print("  --------------------- -------")
    print(f"  BCT pipeline          {counts['bct_flagged']:>5}")
    print(f"  Lexicon baseline      {counts['lexicon_flagged']:>5}")
    print(f"  Stage 3 confirmed     {counts['stage3_confirmed']:>5}")
    print()
    print(f"  Agreement (BCT vs lexicon):")
    print(f"    both:        {agree['both']}")
    print(f"    only BCT:    {agree['only_a']}")
    print(f"    only lex:    {agree['only_b']}")
    print(f"    neither:     {agree['neither']}")
    print(f"    Cohen's kappa: {agree['cohen_kappa']}")
    return 0


# ─── Subparser ─────────────────────────────────────────────────────────────
def add_subparser(sub: argparse._SubParsersAction) -> None:
    eval_p = sub.add_parser("eval", help="Evaluation: baselines, metrics, comparisons.")
    eval_p.add_argument("--corpus-root", default=None,
                        help="Corpus root (default: corpus/)")
    eval_sub = eval_p.add_subparsers(dest="eval_command", required=True)

    lex = eval_sub.add_parser("lexicon", help="Run rule-based lexicon baseline.")
    lex.add_argument("--tender-id", required=True)
    lex.add_argument("--min-markers", type=int, default=1,
                     help="Min number of markers to flag a clause (default 1)")
    lex.set_defaults(func=_cmd_lexicon)

    met = eval_sub.add_parser("metrics", help="Compute precision/recall/F1 + agreement.")
    met.add_argument("--tender-id", required=True)
    met.add_argument("--ground-truth", default=None,
                     help="Path to file with clause_ids (one per line) for "
                          "ground-truth flagged clauses.")
    met.add_argument("--out", default=None,
                     help="Optional path to write JSON metrics dump.")
    met.set_defaults(func=_cmd_metrics)

    cmp = eval_sub.add_parser("compare", help="Side-by-side BCT vs lexicon comparison.")
    cmp.add_argument("--tender-id", required=True)
    cmp.set_defaults(func=_cmd_compare)

    abl = eval_sub.add_parser(
        "ablation",
        help="Per-stage precision/recall ablation against synthetic ground truth.",
    )
    abl.add_argument("--tender-id", required=True,
                     help="Currently only SYN_001 is supported (synthetic GT only).")
    abl.add_argument("--out", default=None,
                     help="Optional path to write the ablation rows as JSON.")
    abl.set_defaults(func=_cmd_ablation)


__all__ = ["add_subparser"]
