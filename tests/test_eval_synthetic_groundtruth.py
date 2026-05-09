"""Tests for the synthetic ground-truth labels."""

from __future__ import annotations

import pytest

from src.eval.synthetic_groundtruth import (
    SYN_001_LABELS,
    SYN_001_PRECISE_CLAUSE_IDS,
    SYN_001_UNIVERSE,
    SYN_001_VAGUE_CLAUSE_IDS,
    category_counts,
    get_ground_truth,
    label_for,
    vague_by_category,
)


def test_universe_partitions_into_vague_and_precise():
    assert SYN_001_VAGUE_CLAUSE_IDS.isdisjoint(SYN_001_PRECISE_CLAUSE_IDS)
    assert SYN_001_UNIVERSE == SYN_001_VAGUE_CLAUSE_IDS | SYN_001_PRECISE_CLAUSE_IDS


def test_some_vague_some_precise():
    """The synthetic tender has BOTH vague and precise clauses by design.
    A degenerate ground truth (all-vague or all-precise) would be unusable."""
    assert len(SYN_001_VAGUE_CLAUSE_IDS) >= 8
    assert len(SYN_001_PRECISE_CLAUSE_IDS) >= 8


def test_known_vague_examples_present():
    """A handful of clauses we know are vague-by-design must be in the set."""
    expected_vague = {
        "01_nit.pdf::00005",   # Eligibility
        "03_scc.pdf::00001",   # Performance security
        "03_scc.pdf::00007",   # Local employment
        "02_gcc.pdf::00003",   # Workmanship
        "04_specs.pdf::00007",  # Aesthetic finish
    }
    assert expected_vague <= SYN_001_VAGUE_CLAUSE_IDS


def test_known_precise_examples_present():
    expected_precise = {
        "01_nit.pdf::00002",   # Estimated cost INR 250 cr
        "04_specs.pdf::00001",  # M30 concrete IS 456:2000
        "02_gcc.pdf::00002",   # Commencement 15/365
    }
    assert expected_precise <= SYN_001_PRECISE_CLAUSE_IDS


def test_label_for_returns_label_or_none():
    g = label_for("03_scc.pdf::00001")
    assert g is not None
    assert g.vague is True
    assert g.category == "delegated_authority"

    assert label_for("nonexistent::id") is None


def test_categories_only_use_documented_buckets():
    """Categories must be one of: precise, subjectivity, effort_pledges,
    delegated_authority, feasibility_dodge, underspecified_quantifier,
    approximation. Catches typos."""
    valid = {
        "precise",
        "subjectivity",
        "effort_pledges",
        "delegated_authority",
        "feasibility_dodge",
        "underspecified_quantifier",
        "approximation",
    }
    for g in SYN_001_LABELS:
        assert g.category in valid, f"{g.clause_id} has unknown category {g.category!r}"


def test_vague_by_category_returns_only_vague():
    cat_set = vague_by_category("delegated_authority")
    for cid in cat_set:
        assert label_for(cid).vague is True
        assert label_for(cid).category == "delegated_authority"


def test_category_counts_sum_to_total_vague():
    counts = category_counts()
    assert sum(counts.values()) == len(SYN_001_VAGUE_CLAUSE_IDS)


def test_get_ground_truth_known_tender():
    gt = get_ground_truth("SYN_001")
    assert gt == SYN_001_VAGUE_CLAUSE_IDS


def test_get_ground_truth_unknown_tender():
    assert get_ground_truth("UNKNOWN_TENDER") is None
