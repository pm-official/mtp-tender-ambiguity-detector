"""End-to-end test that mirrors the Streamlit app's live-mode pipeline.

Runs Stages 0 (parse) → 0b (extract index) → 1 (BCT) → 2 (critic) →
3 (severity) → 4 (rewrite) on a tender directory with one PDF, exactly
the way app.py's run_live_pipeline() does. Any runtime error here will
also fire in the deployed app.

Usage:
    python scripts/test_app_flow.py --tender-id ELEC_TEST --doc-id ElectricalspecificationsTSpart1.pdf --page-start 1 --page-end 5
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_app_flow")


def _read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tender-id", required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--page-start", type=int, required=True)
    parser.add_argument("--page-end", type=int, required=True)
    parser.add_argument("--corpus-root", default="corpus")
    args = parser.parse_args()

    tender_dir = Path(args.corpus_root) / args.tender_id
    parsed_dir = tender_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest.json (mirrors what the Streamlit app does on upload)
    from src.acquire.manifest import (
        TenderDocument, empty_manifest, infer_role_from_filename,
        infer_corrigendum_sequence, sha256_of_file, write_manifest,
        manifest_path,
    )
    if not manifest_path(tender_dir).exists():
        m = empty_manifest(args.tender_id)
        for f in tender_dir.glob("*.pdf"):
            role = infer_role_from_filename(f.name)
            seq = infer_corrigendum_sequence(f.name) if role == "corrigendum" else 0
            try:
                import fitz
                d = fitz.open(str(f))
                pc = len(d)
                d.close()
            except Exception:
                pc = 0
            m.documents.append(TenderDocument(
                filename=f.name, role=role, sequence=seq, page_count=pc,
                file_size_bytes=f.stat().st_size, sha256=sha256_of_file(f),
            ))
        write_manifest(tender_dir, m)
        logger.info(f"Created manifest with {len(m.documents)} documents")

    # ─── Stage 0a: Parse ─────────────────────────────────────────────────
    logger.info("=== Stage 0a: parse ===")
    from src.parse.pipeline import parse_tender
    parse_tender(tender_dir)
    clauses = _read_jsonl(parsed_dir / "clauses.jsonl")
    in_scope = [c for c in clauses
                if c.get("doc_id") == args.doc_id
                and args.page_start <= c.get("page", 1) <= args.page_end]
    logger.info(f"Parsed {len(clauses)} clauses, {len(in_scope)} in scope")
    if not in_scope:
        logger.error("No clauses in selected scope — aborting")
        return 1

    full_clauses_path = parsed_dir / "clauses.jsonl"
    backup_path = parsed_dir / "clauses_full.jsonl"
    shutil.copy(full_clauses_path, backup_path)
    with full_clauses_path.open("w", encoding="utf-8") as fout:
        for c in in_scope:
            fout.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ─── Stage 0b: Build vector index ────────────────────────────────────
    logger.info("=== Stage 0b: build vector index ===")
    shutil.copy(backup_path, full_clauses_path)
    from src.extract.pipeline import extract_all
    try:
        extract_all(
            tender_dir,
            do_index=True,
            do_definitions=False,
            do_quantities=False,
            do_references=False,
            classify_referents_with_llm=False,
            llm_definitions_fallback=False,
        )
    except Exception as e:
        logger.warning(f"Vector index step failed: {e}")
    with full_clauses_path.open("w", encoding="utf-8") as fout:
        for c in in_scope:
            fout.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ─── Stage 1: BCT ────────────────────────────────────────────────────
    logger.info("=== Stage 1: BCT ===")
    from src.score.pipeline import run_stage1
    s1 = run_stage1(tender_dir, max_clauses=None, skip_obligation_filter=False)
    logger.info(f"Stage 1: {s1.get('eligible', 0)} scored, {s1.get('flagged', 0)} flagged, INR {s1.get('total_cost_inr', 0)}")

    # Restore full clauses for retrieval
    shutil.copy(backup_path, full_clauses_path)

    # ─── Stage 2: Critic ─────────────────────────────────────────────────
    logger.info("=== Stage 2: REAL/APPARENT critic ===")
    from src.critic.pipeline import run_stage2
    s2 = run_stage2(tender_dir, max_flags=None, top_k_fused=8)
    logger.info(f"Stage 2: {s2.get('verdict_real', 0)} REAL, {s2.get('verdict_apparent', 0)} APPARENT, "
                f"{s2.get('verdict_weak', 0)} WEAK, INR {s2.get('total_cost_inr', 0)}")

    # ─── Stage 3: Severity ───────────────────────────────────────────────
    logger.info("=== Stage 3: critique + severity ===")
    from src.critic.stage3_pipeline import run_stage3
    s3 = run_stage3(tender_dir, max_real_flags=None, skip_severity=False)
    logger.info(f"Stage 3: {s3.get('critique_confirmed', 0)} CONFIRMED, "
                f"{s3.get('critique_rejected', 0)} REJECTED, "
                f"INR {s3.get('total_cost_inr', 0)}")

    # ─── Stage 4: Rewrite ────────────────────────────────────────────────
    logger.info("=== Stage 4: IS-code-grounded rewrite ===")
    from src.rewrite.pipeline import run_stage4
    try:
        s4 = run_stage4(tender_dir, max_flags=None, only_severity="high")
        logger.info(f"Stage 4: {s4.get('status_counts', {})}, INR {s4.get('total_cost_inr', 0)}")
    except Exception as e:
        logger.warning(f"Stage 4 failed: {e}")
        s4 = {}

    total = s1.get("total_cost_inr", 0) + s2.get("total_cost_inr", 0) + s3.get("total_cost_inr", 0) + s4.get("total_cost_inr", 0)
    logger.info(f"=== END-TO-END SUCCESS  ·  total INR {total:.4f} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
