"""CLI for the IS-code corpus."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.iscode.citation import parse_citations, verify_all_citations
from src.iscode.index import build_iscode_index, index_summary, query_iscode_index


def cmd_build(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[iscode build] Parsing PDFs and embedding new chunks...")
    result = build_iscode_index()
    print(f"\n[iscode build] Done.")
    print(f"  PDFs seen:                {result.pdfs_seen}")
    print(f"  Total chunks across PDFs: {result.chunks_total}")
    print(f"  Newly embedded:           {result.newly_embedded}")
    print(f"  Skipped (already indexed): {result.skipped_already_indexed}")
    print(f"  Embedding cost:           USD {result.cost_usd:.6f}  (~ INR {result.cost_inr:.4f})")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    s = index_summary()
    print("\n[iscode summary]")
    for k, v in s.items():
        print(f"  {k}: {v}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    hits = query_iscode_index(args.query, top_k=args.top_k)
    if not hits:
        print(f"[iscode query] No results.")
        return 1
    print(f"\n[iscode query] Top {len(hits)} for: {args.query!r}\n")
    for i, h in enumerate(hits, 1):
        text = h.text.replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:200] + " ..."
        print(f"  [{i:>2d}] {h.code_id}  §{h.section}  page {h.page}")
        print(f"       {text}\n")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    citations = parse_citations(args.text)
    if not citations:
        print(f"[iscode verify] No citations parsed from input.")
        return 0
    print(f"\n[iscode verify] Found {len(citations)} citation(s):\n")
    verified, resolved, unresolved = verify_all_citations(citations)
    for c in verified:
        mark = "[OK]" if c.resolved else "[--]"
        s = f"§{c.section}" if c.section else "(no section)"
        v = c.version or "any version"
        print(f"  {mark}  {c.code}  {v}  {s}   raw={c.raw_text!r}")
        if not c.resolved:
            print(f"        reason: {c.rejection_reason}")
    print(f"\n  Summary: {resolved} resolved, {unresolved} unresolved")
    return 0


def add_subparser(parent: argparse._SubParsersAction) -> None:
    iscode = parent.add_parser(
        "iscode",
        help="IS-code corpus operations (build / query / verify).",
    )
    sub = iscode.add_subparsers(dest="iscode_command", required=True)

    p = sub.add_parser("build", help="Parse + embed all PDFs in resources/iscode/.")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("summary", help="Show IS-code corpus state.")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("query", help="Top-K semantic query against the IS-code corpus.")
    p.add_argument("query", type=str)
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("verify", help="Parse + verify IS-code citations in a string.")
    p.add_argument("text", type=str, help="Text containing IS-code citations.")
    p.set_defaults(func=cmd_verify)
