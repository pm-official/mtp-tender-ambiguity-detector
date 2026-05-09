"""Offline tests for the BCT deterministic decision rule and schemas."""

from __future__ import annotations

from src.schemas.bct import (
    CANNOT_DETERMINE,
    CONTRACT_DEFINED,
    BCTOutput,
    Commitment,
    evaluate_flag,
)


def _commit(**kwargs):
    """Build a Commitment with sensible defaults for tests."""
    defaults = dict(
        obligation="do something",
        quantity=CANNOT_DETERMINE,
        method=CANNOT_DETERMINE,
        standard=CANNOT_DETERMINE,
        missing_info=[],
    )
    defaults.update(kwargs)
    return Commitment(**defaults)


# ─── Decision rule ──────────────────────────────────────────────────────────
def test_empty_commitments_is_not_flagged():
    bct = BCTOutput(commitments=[])
    flagged, reason = evaluate_flag(bct)
    assert flagged is False
    assert "no commitments" in reason.lower()


def test_all_concrete_is_not_flagged():
    bct = BCTOutput(commitments=[
        _commit(quantity="365 days", method="LoA + 365 days", standard="CONTRACT_DEFINED"),
    ])
    flagged, reason = evaluate_flag(bct)
    assert flagged is False
    assert "all commitments specify" in reason.lower()


def test_missing_quantity_flags():
    bct = BCTOutput(commitments=[
        _commit(quantity=CANNOT_DETERMINE, method="some method", standard="IS 456:2000"),
    ])
    flagged, reason = evaluate_flag(bct)
    assert flagged is True
    assert "quantity" in reason


def test_missing_method_flags():
    bct = BCTOutput(commitments=[
        _commit(quantity="100 mm", method=CANNOT_DETERMINE, standard="IS 456:2000"),
    ])
    flagged, reason = evaluate_flag(bct)
    assert flagged is True
    assert "method" in reason


def test_missing_standard_flags():
    bct = BCTOutput(commitments=[
        _commit(quantity="100 mm", method="lab test", standard=CANNOT_DETERMINE),
    ])
    flagged, reason = evaluate_flag(bct)
    assert flagged is True
    assert "standard" in reason


def test_non_empty_missing_info_flags():
    bct = BCTOutput(commitments=[
        _commit(quantity="100 mm", method="lab", standard="IS 456:2000",
                missing_info=["What kind of lab test?"]),
    ])
    flagged, reason = evaluate_flag(bct)
    assert flagged is True
    assert "missing_info" in reason


def test_contract_defined_does_not_flag_standard():
    bct = BCTOutput(commitments=[
        _commit(quantity="365 days", method="completion certificate", standard=CONTRACT_DEFINED),
    ])
    flagged, reason = evaluate_flag(bct)
    assert flagged is False


def test_multi_commitment_any_flags_whole_clause():
    bct = BCTOutput(commitments=[
        _commit(obligation="o1", quantity="365 days", method="m", standard="s"),
        _commit(obligation="o2", quantity=CANNOT_DETERMINE, method="m", standard="s"),
    ])
    flagged, reason = evaluate_flag(bct)
    assert flagged is True
    assert "o2" in reason or "quantity" in reason


def test_reason_truncates_at_three_obligations():
    bct = BCTOutput(commitments=[
        _commit(obligation=f"obligation_{i}", quantity=CANNOT_DETERMINE)
        for i in range(7)
    ])
    flagged, reason = evaluate_flag(bct)
    assert flagged is True
    assert "+4 more" in reason  # 7 - 3 shown = 4 more


# ─── Schema sanity ──────────────────────────────────────────────────────────
def test_commitment_defaults_to_cannot_determine():
    c = Commitment(obligation="do thing")
    assert c.quantity == CANNOT_DETERMINE
    assert c.method == CANNOT_DETERMINE
    assert c.standard == CANNOT_DETERMINE
    assert c.missing_info == []


def test_bct_output_round_trip():
    bct = BCTOutput(commitments=[
        _commit(obligation="do A", quantity="5%", method="m", standard="s"),
    ])
    payload = bct.model_dump_json()
    restored = BCTOutput.model_validate_json(payload)
    assert restored.commitments[0].obligation == "do A"
    assert restored.commitments[0].quantity == "5%"
