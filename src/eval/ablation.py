"""Ablation runner: precision/recall at each pipeline stage.

For the paper, reviewers want to see how much each pipeline stage
contributes to overall quality. This module reads the artefacts a single
pipeline run already produced and reports precision/recall at four
filtering levels:

  A0  No-filter        — every clause flagged, gives the universe rate.
  A1  Lexicon          — rule-based regex baseline (src/eval/lexicon_baseline.py).
  A2  BCT only         — Stage 1 candidate flags (CANNOT_DETERMINE present).
  A3  + Stage 2 critic — keep only flags Stage 2 marked as REAL.
  A4  + Stage 3 confirm — keep only flags Stage 3 marked as CONFIRMED.

All four positive sets are scored against the same ground truth. The
expected curve is precision climbing A1→A4 while recall holds or drops
slightly — quantifies "more pipeline, fewer false positives".

Output is a list[dict] suitable for printing as a table or writing to JSON
for the paper's ablation table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.eval.metrics import precision_recall


def _load_clause_ids_from_jsonl(
    path: Path, *, where_flagged: bool = True
) -> set[str]:
    """Load the set of clause_ids from a candidate_flags-style JSONL."""
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if where_flagged and not rec.get("flagged", False):
            continue
        out.add(rec["clause_id"])
    return out


def _load_stage2_real_ids(path: Path) -> set[str]:
    """Stage 2: keep clauses with verdict==REAL."""
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("verdict") == "REAL":
            out.add(rec["clause_id"])
    return out


def _load_stage3_confirmed_ids(path: Path) -> set[str]:
    """Stage 3: keep clauses with critique_verdict==CONFIRMED."""
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("critique_verdict") == "CONFIRMED":
            out.add(rec["clause_id"])
    return out


def run_ablation(
    tender_id: str,
    *,
    corpus_root: Optional[Path] = None,
    ground_truth: set[str],
    universe: set[str],
) -> list[dict]:
    """Run the four-rung ablation and return per-rung metrics.

    `ground_truth` is the set of clause_ids that ARE vague; `universe` is
    the full set of clause_ids being evaluated.
    """
    corpus_root = corpus_root or Path("corpus")
    parsed_dir = corpus_root / tender_id / "parsed"

    # Predicted-positive sets at each rung.
    no_filter_set = set(universe)
    lex_set = _load_clause_ids_from_jsonl(parsed_dir / "baseline_lexicon.jsonl")
    bct_set = _load_clause_ids_from_jsonl(parsed_dir / "candidate_flags.jsonl")
    stage2_set = _load_stage2_real_ids(parsed_dir / "stage2_verdicts.jsonl")
    stage3_set = _load_stage3_confirmed_ids(parsed_dir / "stage3_confirmed.jsonl")

    rungs: list[tuple[str, set[str]]] = [
        ("A0_no_filter",       no_filter_set),
        ("A1_lexicon",         lex_set),
        ("A2_bct_stage1",      bct_set),
        ("A3_bct_plus_stage2", stage2_set),
        ("A4_bct_plus_stage3", stage3_set),
    ]

    results: list[dict] = []
    for name, predicted in rungs:
        pr = precision_recall(predicted, ground_truth, universe)
        results.append({
            "rung": name,
            "predicted_positive": len(predicted & universe),
            **pr.to_dict(),
        })
    return results


def format_ablation_table(rows: list[dict]) -> str:
    """Render the ablation result list as a fixed-width table for stdout."""
    headers = ["Rung", "Pred+", "TP", "FP", "FN", "Prec", "Rec", "F1"]
    out = [
        f"{headers[0]:<22}{headers[1]:>7}{headers[2]:>5}"
        f"{headers[3]:>5}{headers[4]:>5}{headers[5]:>8}{headers[6]:>8}{headers[7]:>8}",
        "-" * 70,
    ]
    for r in rows:
        out.append(
            f"{r['rung']:<22}"
            f"{r['predicted_positive']:>7}"
            f"{r['tp']:>5}"
            f"{r['fp']:>5}"
            f"{r['fn']:>5}"
            f"{r['precision']:>8.3f}"
            f"{r['recall']:>8.3f}"
            f"{r['f1']:>8.3f}"
        )
    return "\n".join(out)


__all__ = ["run_ablation", "format_ablation_table"]
