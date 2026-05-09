"""Tests for the ablation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.ablation import format_ablation_table, run_ablation


def _write_jsonl(path: Path, recs: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")


@pytest.fixture
def fake_corpus(tmp_path):
    """Build a fake parsed/ tree with controlled candidate_flags / lexicon /
    Stage 2 / Stage 3 outputs so ablation rungs can be checked."""
    parsed = tmp_path / "TEST_TENDER" / "parsed"
    parsed.mkdir(parents=True)
    # Universe: c1..c10. Vague (positive truth): c1..c5.
    # BCT (Stage 1) flags: c1, c2, c3, c4, c6  (4 TP, 1 FP)
    # Stage 2 REAL: c1, c2, c4, c6           (3 TP, 1 FP — drops c3)
    # Stage 3 CONFIRMED: c1, c2, c4          (3 TP, 0 FP — drops c6)
    # Lexicon: c2, c5, c7                    (2 TP, 1 FP)
    _write_jsonl(parsed / "candidate_flags.jsonl", [
        {"clause_id": cid, "flagged": True} for cid in ["c1", "c2", "c3", "c4", "c6"]
    ] + [
        {"clause_id": cid, "flagged": False} for cid in ["c5", "c7", "c8", "c9", "c10"]
    ])
    _write_jsonl(parsed / "stage2_verdicts.jsonl", [
        {"clause_id": "c1", "verdict": "REAL"},
        {"clause_id": "c2", "verdict": "REAL"},
        {"clause_id": "c3", "verdict": "APPARENT"},
        {"clause_id": "c4", "verdict": "REAL"},
        {"clause_id": "c6", "verdict": "REAL"},
    ])
    _write_jsonl(parsed / "stage3_confirmed.jsonl", [
        {"clause_id": "c1", "critique_verdict": "CONFIRMED"},
        {"clause_id": "c2", "critique_verdict": "CONFIRMED"},
        {"clause_id": "c4", "critique_verdict": "CONFIRMED"},
        {"clause_id": "c6", "critique_verdict": "REJECTED"},
    ])
    _write_jsonl(parsed / "baseline_lexicon.jsonl", [
        {"clause_id": "c2", "flagged": True},
        {"clause_id": "c5", "flagged": True},
        {"clause_id": "c7", "flagged": True},
        {"clause_id": "c1", "flagged": False},
        {"clause_id": "c8", "flagged": False},
    ])
    return tmp_path


def test_ablation_shape(fake_corpus):
    universe = {f"c{i}" for i in range(1, 11)}
    gt = {"c1", "c2", "c3", "c4", "c5"}
    rows = run_ablation(
        tender_id="TEST_TENDER",
        corpus_root=fake_corpus,
        ground_truth=gt,
        universe=universe,
    )
    rung_names = [r["rung"] for r in rows]
    assert rung_names == [
        "A0_no_filter", "A1_lexicon", "A2_bct_stage1",
        "A3_bct_plus_stage2", "A4_bct_plus_stage3",
    ]


def test_ablation_metrics_match_expected(fake_corpus):
    """Expected hand-computed metrics:
      A0 no-filter:     pred=10, tp=5, fp=5, fn=0  → P=0.50, R=1.00, F1=0.667
      A1 lexicon:       pred=3,  tp=2 (c2,c5), fp=1 (c7), fn=3 → P=0.667, R=0.40, F1=0.500
      A2 BCT:           pred=5,  tp=4, fp=1, fn=1 → P=0.80, R=0.80, F1=0.80
      A3 +Stage2 REAL:  pred=4,  tp=3 (c1,c2,c4), fp=1 (c6), fn=2 → P=0.75, R=0.60, F1=0.667
      A4 +Stage3 CONF:  pred=3,  tp=3, fp=0, fn=2 → P=1.00, R=0.60, F1=0.75
    """
    universe = {f"c{i}" for i in range(1, 11)}
    gt = {"c1", "c2", "c3", "c4", "c5"}
    rows = run_ablation(
        tender_id="TEST_TENDER",
        corpus_root=fake_corpus,
        ground_truth=gt,
        universe=universe,
    )
    rung = {r["rung"]: r for r in rows}

    assert rung["A0_no_filter"]["tp"] == 5
    assert rung["A0_no_filter"]["fp"] == 5

    assert rung["A1_lexicon"]["tp"] == 2
    assert rung["A1_lexicon"]["fp"] == 1
    assert abs(rung["A1_lexicon"]["precision"] - 0.6667) < 0.001
    assert rung["A1_lexicon"]["recall"] == pytest.approx(0.40)

    assert rung["A2_bct_stage1"]["tp"] == 4
    assert rung["A2_bct_stage1"]["fp"] == 1
    assert rung["A2_bct_stage1"]["precision"] == pytest.approx(0.80)
    assert rung["A2_bct_stage1"]["recall"] == pytest.approx(0.80)

    assert rung["A3_bct_plus_stage2"]["tp"] == 3
    assert rung["A3_bct_plus_stage2"]["fp"] == 1
    assert rung["A3_bct_plus_stage2"]["precision"] == pytest.approx(0.75)

    # Stage 3 raises precision to 1.0 — the headline ablation result.
    assert rung["A4_bct_plus_stage3"]["tp"] == 3
    assert rung["A4_bct_plus_stage3"]["fp"] == 0
    assert rung["A4_bct_plus_stage3"]["precision"] == pytest.approx(1.0)


def test_format_ablation_table_runs(fake_corpus):
    universe = {f"c{i}" for i in range(1, 11)}
    gt = {"c1", "c2", "c3", "c4", "c5"}
    rows = run_ablation(
        tender_id="TEST_TENDER",
        corpus_root=fake_corpus,
        ground_truth=gt,
        universe=universe,
    )
    s = format_ablation_table(rows)
    assert "A0_no_filter" in s
    assert "A4_bct_plus_stage3" in s
    assert "Prec" in s
    # Five rungs + header lines
    assert s.count("\n") >= 6
