"""CLI sub-commands for Session 4 — vector index, definitions, quantities, references.

Wired into `python -m src.cli extract ...` via src/cli.py.

Sub-commands:
  run        — run all extractors (or pick subsets via flags)
  query      — query the vector index for top-K similar clauses
  summary    — print a summary of what's been extracted for a tender
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.extract.pipeline import extract_all
from src.extract.vector_index import collection_summary, query_vector_index


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
        print(f"[extract run] {folder} does not exist.")
        return 1

    print(f"[extract run] Extracting structures for {args.tender_id}...")
    summary = extract_all(
        folder,
        do_index=not args.no_index,
        do_definitions=not args.no_definitions,
        do_quantities=not args.no_quantities,
        do_references=not args.no_references,
        classify_referents_with_llm=args.classify_referents,
        llm_definitions_fallback=args.llm_definitions,
    )

    print("\n[extract run] Done.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    hits = query_vector_index(
        args.query,
        tender_id=args.tender_id,
        top_k=args.top_k,
    )
    if not hits:
        print(f"[extract query] No results (collection may be empty or missing).")
        return 1
    print(f"\n[extract query] Top {len(hits)} for {args.tender_id}:\n")
    for h in hits:
        meta = h.metadata or {}
        section = meta.get("section", "")
        clause_num = meta.get("clause_number") or "-"
        page = meta.get("page", "?")
        text = h.text.replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:200] + " ..."
        print(f"  [{h.rank:>2d}] score={h.score:.3f}  page {page}  clause={clause_num:<10} section={section[:40]}")
        print(f"       {text}\n")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    if not folder.exists():
        print(f"[extract summary] {folder} does not exist.")
        return 1

    parsed = folder / "parsed"
    print(f"\n[extract summary] {args.tender_id}")
    print(f"  parsed dir: {parsed}")

    for name in ("clauses.jsonl", "definitions.jsonl", "quantities.jsonl", "references.jsonl"):
        p = parsed / name
        if not p.exists():
            print(f"  - {name:<20} (missing)")
            continue
        nlines = sum(1 for _ in p.open("r", encoding="utf-8"))
        print(f"  - {name:<20} {nlines:>6d} entries")

    rg = parsed / "reference_graph.json"
    if rg.exists():
        with rg.open("r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = len(data.get("nodes") or [])
        edges = len(data.get("edges") or data.get("links") or [])
        print(f"  - reference_graph.json   nodes={nodes} edges={edges}")
    else:
        print("  - reference_graph.json (missing)")

    cs = collection_summary(args.tender_id)
    if cs.get("exists"):
        print(f"  - vector index           {cs['count']} embeddings (collection {cs['name']})")
    else:
        print("  - vector index           (none)")
    return 0


def add_subparser(parent: argparse._SubParsersAction) -> None:
    extract = parent.add_parser(
        "extract",
        help="Stage 0 part 2 — vector index + definitions + quantities + references.",
    )
    extract.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Override the default corpus/ directory.",
    )
    sub = extract.add_subparsers(dest="extract_command", required=True)

    p = sub.add_parser("run", help="Run all extractors on a tender.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("--no-index", action="store_true", help="Skip ChromaDB embedding step.")
    p.add_argument("--no-definitions", action="store_true")
    p.add_argument("--no-quantities", action="store_true")
    p.add_argument("--no-references", action="store_true")
    p.add_argument("--classify-referents", action="store_true",
                    help="Use Gemini Flash to assign canonical referent labels to quantities.")
    p.add_argument("--llm-definitions", action="store_true",
                    help="Use Gemini Flash fallback for Definitions sections that produced no regex hits.")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("query", help="Query the vector index for top-K similar clauses.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("query", type=str)
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("summary", help="Show what's been extracted for a tender.")
    p.add_argument("--tender-id", required=True)
    p.set_defaults(func=cmd_summary)
