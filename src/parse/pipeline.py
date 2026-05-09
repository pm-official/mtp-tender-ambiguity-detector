"""End-to-end Stage-0 parsing for one tender.

Reads corpus/<tender_id>/manifest.json, parses every PDF it lists, runs the
clause-aware chunker, runs table extraction, and writes:

  corpus/<tender_id>/parsed/document_tree.json   per-document parse summary
  corpus/<tender_id>/parsed/clauses.jsonl        one Clause per line, all docs

Ordering: documents are processed in manifest order; clause IDs are unique
within a doc, so the global clauses.jsonl can be sliced by doc_id.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.acquire.manifest import read_manifest
from src.parse.chunker import chunk_document
from src.parse.parser import parse_pdf
from src.parse.tables import extract_tables_as_clauses
from src.schemas.clause import Clause, DocumentTree

logger = logging.getLogger(__name__)


def parse_tender(
    tender_dir: Path,
    *,
    enable_ocr: bool = True,
    enable_tables: bool = True,
) -> dict:
    """Parse all PDFs in a tender directory. Returns summary dict."""
    tender_dir = Path(tender_dir)
    manifest = read_manifest(tender_dir)
    out_dir = tender_dir / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_clauses: list[Clause] = []
    document_trees: list[DocumentTree] = []

    for doc in manifest.documents:
        pdf_path = tender_dir / doc.filename
        if not pdf_path.exists():
            logger.warning("PDF missing on disk: %s", pdf_path)
            continue

        logger.info("Parsing %s (%d pages, role=%s)...",
                    doc.filename, doc.page_count, doc.role)

        parsed = parse_pdf(
            pdf_path,
            tender_id=manifest.tender_id,
            doc_id=doc.filename,
            enable_ocr=enable_ocr,
        )

        clauses = chunk_document(parsed)
        if enable_tables:
            table_clauses = extract_tables_as_clauses(
                parsed,
                starting_clause_index=len(clauses),
            )
        else:
            table_clauses = []

        all_in_doc = clauses + table_clauses

        tree = DocumentTree(
            doc_id=doc.filename,
            tender_id=manifest.tender_id,
            page_count=parsed.page_count,
            pages=parsed.page_info_list(),
            clause_count=len(all_in_doc),
            table_clause_count=len(table_clauses),
            ocr_pages=parsed.ocr_pages,
        )
        document_trees.append(tree)
        all_clauses.extend(all_in_doc)

    # Write outputs
    tree_path = out_dir / "document_tree.json"
    with tree_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tender_id": manifest.tender_id,
                "parsed_at": datetime.now(timezone.utc).isoformat(),
                "documents": [t.model_dump(mode="json") for t in document_trees],
                "summary": {
                    "total_clauses": len(all_clauses),
                    "table_clauses": sum(1 for c in all_clauses if c.table_origin),
                    "ocr_pages_total": sum(len(t.ocr_pages) for t in document_trees),
                    "documents_parsed": len(document_trees),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    clauses_path = out_dir / "clauses.jsonl"
    with clauses_path.open("w", encoding="utf-8") as f:
        for c in all_clauses:
            f.write(c.model_dump_json() + "\n")

    return {
        "tender_id": manifest.tender_id,
        "documents_parsed": len(document_trees),
        "total_clauses": len(all_clauses),
        "table_clauses": sum(1 for c in all_clauses if c.table_origin),
        "ocr_pages_total": sum(len(t.ocr_pages) for t in document_trees),
        "tree_path": str(tree_path),
        "clauses_path": str(clauses_path),
    }
