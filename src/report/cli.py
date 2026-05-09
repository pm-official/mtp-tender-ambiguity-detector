"""CLI for the report renderer."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from src.report.render import render_html_report


def _project_corpus_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent / "corpus"
    return Path.cwd() / "corpus"


def cmd_html(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    if not folder.exists():
        print(f"[report html] {folder} does not exist.")
        return 1

    out = render_html_report(folder)
    print(f"[report html] Wrote {out}")
    if args.open:
        webbrowser.open(out.as_uri())
    return 0


def add_subparser(parent: argparse._SubParsersAction) -> None:
    report = parent.add_parser("report", help="Render user-facing reports.")
    report.add_argument("--corpus-root", type=Path, default=None)
    sub = report.add_subparsers(dest="report_command", required=True)

    p = sub.add_parser("html", help="Render the tender's HTML report.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("--open", action="store_true", help="Open the report in your default browser when done.")
    p.set_defaults(func=cmd_html)
