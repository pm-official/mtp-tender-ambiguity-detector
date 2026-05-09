"""Tests for the eval metrics module."""

from __future__ import annotations

import pytest

from src.eval.metrics import (
    PrecisionRecall,
    agreement,
    cohens_kappa,
    precision_recall,
)


# ─── Precision / recall / F1 ───────────────────────────────────────────────
def test_precision_recall_perfect():
    pr = precision_recall(
        predicted_positive={"a", "b", "c"},
        ground_truth_positive={"a", "b", "c"},
        universe={"a", "b", "c", "d", "e"},
    )
    assert pr.precision == 1.0
    assert pr.recall == 1.0
    assert pr.f1 == 1.0
    assert pr.tp == 3
    assert pr.fp == 0
    assert pr.fn == 0
    assert pr.tn == 2


def test_precision_recall_half():
    pr = precision_recall(
        predicted_positive={"a", "b", "x", "y"},   # 2 right, 2 wrong
        ground_truth_positive={"a", "b", "c", "d"},  # 4 true
        universe={"a", "b", "c", "d", "x", "y", "z"},
    )
    assert pr.tp == 2
    assert pr.fp == 2
    assert pr.fn == 2
    assert pr.precision == 0.5
    assert pr.recall == 0.5
    assert pr.f1 == 0.5


def test_precision_recall_no_predictions():
    pr = precision_recall(
        predicted_positive=set(),
        ground_truth_positive={"a", "b"},
        universe={"a", "b", "c"},
    )
    assert pr.precision == 0.0
    assert pr.recall == 0.0
    assert pr.f1 == 0.0


def test_precision_recall_no_ground_truth():
    pr = precision_recall(
        predicted_positive={"a"},
        ground_truth_positive=set(),
        universe={"a", "b", "c"},
    )
    assert pr.recall == 0.0
    assert pr.precision == 0.0   # tp=0, fp=1, denom 1, precision 0
    assert pr.fn == 0


def test_precision_recall_to_dict():
    pr = PrecisionRecall(tp=1, fp=1, fn=0, tn=1,
                          precision=0.5, recall=1.0, f1=0.6667, accuracy=0.6667)
    d = pr.to_dict()
    assert d["tp"] == 1
    assert d["precision"] == 0.5


# ─── Cohen's kappa ─────────────────────────────────────────────────────────
def test_cohens_kappa_perfect_agreement():
    k = cohens_kappa(both=10, only_a=0, only_b=0, neither=10)
    assert k == 1.0


def test_cohens_kappa_zero_when_random():
    """Two independent flippers with 50% positive rate → expected κ ≈ 0."""
    # 25/25/25/25 grid → Po=0.5, Pe=0.5 → κ=0
    k = cohens_kappa(both=25, only_a=25, only_b=25, neither=25)
    assert abs(k) < 1e-9


def test_cohens_kappa_negative_for_anti_correlation():
    # Pure anti-correlation → κ < 0
    k = cohens_kappa(both=0, only_a=10, only_b=10, neither=0)
    assert k < 0


# ─── Agreement matrix ──────────────────────────────────────────────────────
def test_agreement_basic():
    am = agreement(
        flagged_a={"a", "b", "c"},
        flagged_b={"b", "c", "d"},
        universe={"a", "b", "c", "d", "e"},
        name_a="bct",
        name_b="lex",
    )
    assert am.both == 2          # b, c
    assert am.only_a == 1        # a
    assert am.only_b == 1        # d
    assert am.neither == 1       # e
    assert am.name_a == "bct"


def test_agreement_universe_filtering():
    """Items outside the universe should be ignored entirely."""
    am = agreement(
        flagged_a={"a", "extra"},
        flagged_b={"a", "another_extra"},
        universe={"a", "b", "c"},
        name_a="bct",
        name_b="lex",
    )
    assert am.both == 1   # a (extras filtered out)
    assert am.only_a == 0
    assert am.only_b == 0
    assert am.neither == 2   # b, c
