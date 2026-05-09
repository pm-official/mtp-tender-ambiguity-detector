"""Evaluation metrics: precision/recall/F1 + agreement matrices.

Two flavours supported:

  1. Reference-based: precision/recall/F1 against a ground-truth set of
     flagged clause_ids (e.g. SYN_001 has known intentionally-vague clauses).
  2. Reference-free agreement: confusion matrix between the BCT pipeline
     output and the lexicon baseline. Useful when no ground truth exists
     (e.g. JK_001 real tender) — tells you which clauses each method picks
     up that the other misses, which is itself diagnostically useful.

All metrics are at the **clause** level, keyed by clause_id (or flag_id's
clause portion).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# ─── Helpers ───────────────────────────────────────────────────────────────
def _load_flag_ids(path: Path, *, key: str = "clause_id",
                    require_flagged: bool = True) -> set[str]:
    """Load the set of clause_ids flagged in a candidate_flags-style JSONL."""
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if require_flagged and not rec.get("flagged", False):
            continue
        out.add(rec[key])
    return out


def _load_confirmed_ids(path: Path) -> set[str]:
    """Load the set of clause_ids with critique_verdict==CONFIRMED in stage3."""
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


# ─── Reference-based metrics ───────────────────────────────────────────────
@dataclass
class PrecisionRecall:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


def precision_recall(
    predicted_positive: set[str],
    ground_truth_positive: set[str],
    universe: Iterable[str],
) -> PrecisionRecall:
    """Compute precision, recall, F1 and accuracy at the clause level.

    `universe` is the set of all clause_ids considered (so we can compute TN).
    """
    universe_set = set(universe)
    pp = predicted_positive & universe_set
    gt = ground_truth_positive & universe_set
    tp = len(pp & gt)
    fp = len(pp - gt)
    fn = len(gt - pp)
    tn = len(universe_set - pp - gt)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(universe_set), 1)

    return PrecisionRecall(
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1, accuracy=accuracy,
    )


# ─── Reference-free agreement (BCT vs lexicon) ─────────────────────────────
@dataclass
class AgreementMatrix:
    """Confusion matrix when neither method is ground truth.

    Cells:
      both    — flagged by both methods
      only_a  — flagged only by method A
      only_b  — flagged only by method B
      neither — flagged by neither
    """
    both: int
    only_a: int
    only_b: int
    neither: int
    cohen_kappa: float
    name_a: str
    name_b: str

    def to_dict(self) -> dict:
        return {
            "name_a": self.name_a, "name_b": self.name_b,
            "both": self.both, "only_a": self.only_a,
            "only_b": self.only_b, "neither": self.neither,
            "cohen_kappa": round(self.cohen_kappa, 4),
        }


def cohens_kappa(both: int, only_a: int, only_b: int, neither: int) -> float:
    """Cohen's κ for two binary classifiers on the same items."""
    n = both + only_a + only_b + neither
    if n == 0:
        return 0.0
    po = (both + neither) / n
    pa_pos = (both + only_a) / n
    pb_pos = (both + only_b) / n
    pe = pa_pos * pb_pos + (1 - pa_pos) * (1 - pb_pos)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def agreement(
    flagged_a: set[str],
    flagged_b: set[str],
    universe: Iterable[str],
    name_a: str = "A",
    name_b: str = "B",
) -> AgreementMatrix:
    universe_set = set(universe)
    a = flagged_a & universe_set
    b = flagged_b & universe_set
    both = len(a & b)
    only_a = len(a - b)
    only_b = len(b - a)
    neither = len(universe_set - a - b)
    k = cohens_kappa(both, only_a, only_b, neither)
    return AgreementMatrix(
        both=both, only_a=only_a, only_b=only_b, neither=neither,
        cohen_kappa=k, name_a=name_a, name_b=name_b,
    )


# ─── Top-level eval driver ─────────────────────────────────────────────────
def eval_tender(
    tender_id: str,
    *,
    corpus_root: Optional[Path] = None,
    ground_truth: Optional[set[str]] = None,
    universe_override: Optional[set[str]] = None,
) -> dict:
    """Compute every available metric for one tender.

    Returns a dict with keys:
      - 'bct_pipeline.precision_recall'   (only if ground_truth given)
      - 'bct_pipeline.confirmed.precision_recall' (only if ground_truth given,
            uses Stage 3 CONFIRMED set)
      - 'lexicon_baseline.precision_recall' (only if ground_truth given)
      - 'agreement.bct_vs_lexicon' (always — reference-free)
    Plus per-method clause counts.

    If `universe_override` is given, restrict the evaluation universe to that
    set (typically the union of vague + precise hand-labelled clauses for a
    synthetic tender). Without it, the universe is the union of clauses any
    method observed — which inflates TN counts with document-header lines
    and tables that aren't real clauses.
    """
    corpus_root = corpus_root or Path("corpus")
    parsed_dir = corpus_root / tender_id / "parsed"

    if universe_override is not None:
        universe = set(universe_override)
    else:
        universe = _load_flag_ids(
            parsed_dir / "candidate_flags.jsonl", require_flagged=False,
        ) | _load_flag_ids(
            parsed_dir / "baseline_lexicon.jsonl", require_flagged=False,
        )

    bct_set = _load_flag_ids(parsed_dir / "candidate_flags.jsonl")
    lex_set = _load_flag_ids(parsed_dir / "baseline_lexicon.jsonl")
    confirmed_set = _load_confirmed_ids(parsed_dir / "stage3_confirmed.jsonl")

    out: dict = {
        "tender_id": tender_id,
        "universe_size": len(universe),
        "counts": {
            "bct_flagged": len(bct_set),
            "lexicon_flagged": len(lex_set),
            "stage3_confirmed": len(confirmed_set),
        },
        "agreement": {
            "bct_vs_lexicon": agreement(
                bct_set, lex_set, universe,
                name_a="bct_pipeline", name_b="lexicon_baseline",
            ).to_dict(),
        },
    }

    if ground_truth is not None:
        gt = set(ground_truth)
        out["ground_truth_size"] = len(gt)
        out["bct_pipeline"] = {
            "precision_recall": precision_recall(bct_set, gt, universe).to_dict(),
        }
        if confirmed_set:
            out["bct_confirmed"] = {
                "precision_recall": precision_recall(
                    confirmed_set, gt, universe
                ).to_dict(),
            }
        out["lexicon_baseline"] = {
            "precision_recall": precision_recall(lex_set, gt, universe).to_dict(),
        }

    return out


__all__ = [
    "PrecisionRecall",
    "AgreementMatrix",
    "precision_recall",
    "agreement",
    "cohens_kappa",
    "eval_tender",
]
