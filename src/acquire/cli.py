"""CLI sub-commands for tender acquisition.

Wired into `python -m src.cli acquire ...` via src/cli.py.

Sub-commands:
  list       — list every tender in corpus/, with manifest summaries
  init       — create an empty corpus/<tender_id>/ folder + manifest stub
  drop       — given a folder you've manually filled with PDFs, build a manifest
  validate   — check an existing tender folder is well-formed
  synthesize — generate a synthetic tender for development testing
  scrape     — open Chrome via Selenium and semi-automatically download from CPPP
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.acquire.manifest import (
    Manifest,
    TenderDocument,
    empty_manifest,
    infer_corrigendum_sequence,
    infer_role_from_filename,
    manifest_path,
    read_manifest,
    sha256_of_file,
    write_manifest,
)
from src.acquire.pdf_validator import inspect_pdf


def _project_corpus_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent / "corpus"
    return Path.cwd() / "corpus"


# ─── list ───────────────────────────────────────────────────────────────────
def cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    if not root.exists():
        print(f"[acquire list] Corpus root {root} does not exist.")
        return 1

    rows: list[tuple[str, str, int, int, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        mp = manifest_path(child)
        if not mp.exists():
            rows.append((child.name, "(no manifest.json)", 0, 0, ""))
            continue
        try:
            m = read_manifest(child)
            rows.append((
                m.tender_id,
                m.issuing_authority,
                m.num_documents,
                m.num_corrigenda,
                m.retrieved_at.isoformat(timespec="seconds"),
            ))
        except Exception as e:
            rows.append((child.name, f"(invalid manifest: {e})", 0, 0, ""))

    if not rows:
        print(f"[acquire list] No tender folders in {root}.")
        return 0

    print(f"\nTender corpus at {root}\n")
    print(f"  {'tender_id':<28} {'authority':<35} {'#docs':>5} {'#corr':>5}   retrieved_at")
    print(f"  {'-'*28} {'-'*35} {'-'*5} {'-'*5}   {'-'*20}")
    for tid, auth, ndocs, ncorr, when in rows:
        print(f"  {tid:<28} {auth[:35]:<35} {ndocs:>5} {ncorr:>5}   {when}")
    return 0


# ─── init ───────────────────────────────────────────────────────────────────
def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    if folder.exists() and any(folder.iterdir()):
        print(f"[acquire init] {folder} already exists and is not empty. Aborting.")
        return 1

    folder.mkdir(parents=True, exist_ok=True)
    m = empty_manifest(args.tender_id)
    if args.authority:
        m.issuing_authority = args.authority
    if args.title:
        m.project_title = args.title
    if args.source_url:
        m.source_url = args.source_url
    write_manifest(folder, m)

    print(f"[acquire init] Created {folder}")
    print(f"  manifest stub written to {manifest_path(folder)}")
    print("  Next: drop your tender PDFs into this folder and run:")
    print(f"    python -m src.cli acquire drop --tender-id {args.tender_id}")
    return 0


# ─── drop ───────────────────────────────────────────────────────────────────
def cmd_drop(args: argparse.Namespace) -> int:
    """User has manually downloaded PDFs into corpus/<tender_id>/. Build the
    manifest from filenames + PDF inspection."""
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    if not folder.exists():
        print(f"[acquire drop] {folder} does not exist. Run `init` first.")
        return 1

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"[acquire drop] No PDFs in {folder}. Drop your tender PDFs in and re-run.")
        return 1

    # Try to load existing manifest stub for tender-level metadata
    if (folder / "manifest.json").exists():
        try:
            m = read_manifest(folder)
        except Exception:
            m = empty_manifest(args.tender_id)
    else:
        m = empty_manifest(args.tender_id)

    # Always rebuild the documents list
    documents: list[TenderDocument] = []
    print(f"\n[acquire drop] Inspecting {len(pdfs)} PDFs in {folder}...")
    for pdf in pdfs:
        info = inspect_pdf(pdf, sample_text_layer=True)
        if not info.is_pdf:
            print(f"  [WARN] {pdf.name}: not a valid PDF ({info.error})")
            continue
        role = infer_role_from_filename(pdf.name)
        seq = infer_corrigendum_sequence(pdf.name) if role == "corrigendum" else 0
        documents.append(
            TenderDocument(
                filename=pdf.name,
                role=role,
                sequence=seq,
                page_count=info.page_count,
                file_size_bytes=info.file_size_bytes,
                sha256=sha256_of_file(pdf),
                notes=info.error or "",
            )
        )
        flag = "[OCR-only]" if not info.has_text_layer else ""
        print(f"  {pdf.name:<40}  role={role:<12} pages={info.page_count:>4}  {flag}")

    m.documents = documents
    m.retrieved_at = datetime.now(tz=timezone.utc)
    write_manifest(folder, m)

    print(f"\n[acquire drop] Manifest updated:")
    print(f"  {len(documents)} documents tagged ({m.num_corrigenda} corrigenda).")
    print(f"  Edit {manifest_path(folder)} if any role looks wrong.")
    return 0


# ─── validate ───────────────────────────────────────────────────────────────
def cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root or _project_corpus_root())
    folder = root / args.tender_id
    if not folder.exists():
        print(f"[acquire validate] {folder} does not exist.")
        return 1

    try:
        m = read_manifest(folder)
    except Exception as e:
        print(f"[acquire validate] Manifest invalid: {e}")
        return 2

    issues: list[str] = []
    # 1. Every document referenced by manifest must exist on disk
    for d in m.documents:
        p = folder / d.filename
        if not p.exists():
            issues.append(f"manifest references missing file: {d.filename}")
            continue
        actual_size = p.stat().st_size
        if actual_size != d.file_size_bytes and d.file_size_bytes > 0:
            issues.append(
                f"{d.filename}: size mismatch (manifest={d.file_size_bytes}, disk={actual_size})"
            )
        if d.sha256:
            actual_sha = sha256_of_file(p)
            if actual_sha != d.sha256:
                issues.append(f"{d.filename}: SHA256 mismatch (file changed since manifest)")

    # 2. Every PDF on disk must be in the manifest
    on_disk = {p.name for p in folder.glob("*.pdf")}
    in_manifest = {d.filename for d in m.documents}
    extra = on_disk - in_manifest
    for f in sorted(extra):
        issues.append(f"PDF on disk but missing from manifest: {f}")

    print(f"\n[acquire validate] Tender: {m.tender_id}")
    print(f"  Authority:    {m.issuing_authority}")
    print(f"  Documents:    {m.num_documents}  (corrigenda: {m.num_corrigenda})")
    print(f"  Retrieved at: {m.retrieved_at.isoformat(timespec='seconds')}")
    if not issues:
        print("  Validation:   OK")
        return 0
    print(f"  Validation:   {len(issues)} issue(s):")
    for i in issues:
        print(f"    - {i}")
    return 3


# ─── synthesize ─────────────────────────────────────────────────────────────
def cmd_synthesize(args: argparse.Namespace) -> int:
    from src.acquire.synthetic import generate_synthetic_tender

    root = Path(args.corpus_root or _project_corpus_root())
    folder = generate_synthetic_tender(
        out_root=root,
        tender_id=args.tender_id,
        overwrite=args.overwrite,
    )
    print(f"[acquire synthesize] Wrote synthetic tender to {folder}")
    return 0


# ─── scrape (Selenium) ──────────────────────────────────────────────────────
def cmd_scrape(args: argparse.Namespace) -> int:
    from src.acquire.cppp_scraper import semi_automatic_download

    root = Path(args.corpus_root or _project_corpus_root())
    folder = semi_automatic_download(
        out_root=root,
        tender_id=args.tender_id,
        initial_url=args.url,
        headless=False,
    )
    print(f"[acquire scrape] Done. Tender folder: {folder}")
    return 0


# ─── Argparse wiring ────────────────────────────────────────────────────────
def add_subparser(parent: argparse._SubParsersAction) -> None:
    """Attach `acquire` subcommands to the top-level CLI."""
    acquire = parent.add_parser(
        "acquire",
        help="Acquire and validate tender packages.",
    )
    acquire.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Override the default corpus/ directory.",
    )
    sub = acquire.add_subparsers(dest="acquire_command", required=True)

    p = sub.add_parser("list", help="List tenders in corpus/.")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("init", help="Create an empty tender folder + manifest stub.")
    p.add_argument("--tender-id", required=True)
    p.add_argument("--authority", default="")
    p.add_argument("--title", default="")
    p.add_argument("--source-url", default="")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("drop", help="Build manifest from PDFs already in folder.")
    p.add_argument("--tender-id", required=True)
    p.set_defaults(func=cmd_drop)

    p = sub.add_parser("validate", help="Validate an existing tender folder.")
    p.add_argument("--tender-id", required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("synthesize", help="Generate a synthetic tender for development.")
    p.add_argument("--tender-id", default="SYN_001")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_synthesize)

    p = sub.add_parser("scrape", help="Selenium-driven semi-automatic CPPP downloader.")
    p.add_argument("--tender-id", required=True)
    p.add_argument(
        "--url",
        default="https://etenders.gov.in/eprocure/app",
        help="Initial URL to navigate to (default: CPPP home).",
    )
    p.set_defaults(func=cmd_scrape)
