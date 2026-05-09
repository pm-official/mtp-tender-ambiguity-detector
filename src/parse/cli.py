"""CLI sub-commands for parsing tender PDFs.

Wired into `python -m src.cli parse ...` via src/cli.py.

Sub-commands:
  run        — parse all PDFs in corpus/<tender_id>/, write clauses.jsonl
  inspect    — print N random clauses from a parsed tender for spot-checking
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.parse.pipeline import parse_tender


def _project_corpus_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent / "corpus"
    return Path.cwd() / "corpus"


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    if not folder.exists():
        print(f"[parse run] {folder} does not exist.")
        return 1

    print(f"[parse run] Parsing tender {args.tender_id}...")
    result = parse_tender(
        folder,
        enable_ocr=not args.no_ocr,
        enable_tables=not args.no_tables,
    )

    print(f"\n[parse run] Done.")
    print(f"  Documents parsed:   {result['documents_parsed']}")
    print(f"  Total clauses:      {result['total_clauses']}")
    print(f"  Table-row clauses:  {result['table_clauses']}")
    print(f"  Pages requiring OCR:{result['ocr_pages_total']}")
    print(f"  Output:")
    print(f"    {result['tree_path']}")
    print(f"    {result['clauses_path']}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    clauses_path = folder / "parsed" / "clauses.jsonl"
    if not clauses_path.exists():
        print(f"[parse inspect] {clauses_path} not found. Run `parse run` first.")
        return 1

    rows: list[dict] = []
    with clauses_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        print(f"[parse inspect] {clauses_path} is empty.")
        return 0

    n = min(args.n, len(rows))
    if args.role:
        filtered = [r for r in rows if (
            r.get("table_origin", False) if args.role == "table" else True
        )]
        rows = filtered or rows

    rng = random.Random(args.seed)
    sample = rng.sample(rows, n) if n <= len(rows) else rows

    print(f"\n[parse inspect] {n} of {len(rows)} clauses from {args.tender_id}:\n")
    for i, c in enumerate(sample, 1):
        section = " > ".join(c.get("section_path", []) or ["(none)"])
        flags = []
        if c.get("table_origin"):
            flags.append("TABLE")
        if c.get("ocr_was_used"):
            flags.append("OCR")
        flag_str = ("  [" + ",".join(flags) + "]") if flags else ""
        head = (
            f"  [{i:>2d}] {c['doc_id']}  page {c['page']}  "
            f"clause={c.get('clause_number') or '-':<10}  "
            f"chars={c['char_count']:>5}  section={section[:40]}{flag_str}"
        )
        print(head)
        text = c["text"].replace("\n", " ").strip()
        if len(text) > 280:
            text = text[:280] + " ..."
        print(f"       {text}\n")
    return 0


def add_subparser(parent: argparse._SubParsersAction) -> None:
    parse = parent.add_parser(
        "parse",
        help="Stage 0 — parse tender PDFs into clauses.jsonl.",
    )
    parse.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Override the default corpus/ directory.",
    )
    sub = parse.add_subparsers(dest="parse_command", required=True)

    p = sub.add_parser("run", help="Parse all PDFs in corpus/<tender_id>/.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("--no-ocr", action="store_true", help="Skip OCR fallback for scanned pages.")
    p.add_argument("--no-tables", action="store_true", help="Skip table extraction.")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("inspect", help="Spot-check N random clauses.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("-n", type=int, default=15, help="How many clauses to sample (default 15).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--role", choices=["any", "table"], default="any")
    p.set_defaults(func=cmd_inspect)
