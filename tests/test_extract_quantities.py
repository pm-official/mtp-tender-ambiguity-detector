"""Offline tests for the quantities extractor."""

from __future__ import annotations

import pytest

from src.extract.quantities import _normalise, extract_quantities
from src.schemas.clause import Clause


def _clause(text: str, *, clause_id="c1") -> Clause:
    return Clause(
        clause_id=clause_id,
        doc_id="doc.pdf",
        tender_id="T1",
        page=1,
        page_end=1,
        section_path=[],
        clause_number=None,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
    )


# ─── Regex extraction ──────────────────────────────────────────────────────
def test_extract_duration_in_days():
    qs = extract_quantities([_clause("complete within 365 days of commencement.")])
    assert any(q.kind == "duration" and q.value == 365 and q.unit_norm == "days" for q in qs)


def test_extract_duration_in_months_normalised():
    qs = extract_quantities([_clause("the period shall be 6 months from LoA.")])
    duration = next((q for q in qs if q.kind == "duration"), None)
    assert duration is not None
    assert duration.value == 6
    # 6 months = 180 days canonical
    assert duration.value_norm == 180
    assert duration.unit_norm == "days"


def test_extract_percentage():
    qs = extract_quantities([_clause("equal to 5% of the Contract Price.")])
    assert any(q.kind == "percentage" and q.value == 5 and q.unit_norm == "pct" for q in qs)


def test_extract_money_lakh():
    qs = extract_quantities([_clause("Rs. 50 lakh deposit shall be furnished.")])
    money = next((q for q in qs if q.kind == "money"), None)
    assert money is not None
    assert money.value == 50
    # 50 lakh in canonical inr_lakhs = 50
    assert money.value_norm == 50
    assert money.unit_norm == "inr_lakhs"


def test_extract_money_crore():
    qs = extract_quantities([_clause("an estimated cost of INR 250 crore.")])
    money = next((q for q in qs if q.kind == "money"), None)
    assert money is not None
    # 250 crore = 25,000 lakh
    assert money.value_norm == 25_000


def test_extract_dimension_mm():
    qs = extract_quantities([_clause("slump 75 mm at the point of placement.")])
    dim = next((q for q in qs if q.kind == "dimension"), None)
    assert dim is not None
    assert dim.value == 75
    assert dim.value_norm == 75
    assert dim.unit_norm == "mm"


def test_extract_dimension_metres_normalised():
    qs = extract_quantities([_clause("clearance shall be 2.5 m above ground.")])
    dim = next((q for q in qs if q.kind == "dimension"), None)
    assert dim is not None
    # 2.5 m = 2500 mm
    assert dim.value_norm == 2500


def test_extract_tolerance_pm():
    qs = extract_quantities([_clause("tolerance ± 2 mm in finished surface.")])
    tol = next((q for q in qs if q.kind == "tolerance"), None)
    assert tol is not None
    assert tol.value == 2


def test_extract_multiple_quantities_in_one_clause():
    text = "Slump 75 mm and complete within 365 days, with deposit Rs. 50 lakh."
    qs = extract_quantities([_clause(text)])
    kinds = {q.kind for q in qs}
    assert "duration" in kinds
    assert "money" in kinds
    assert "dimension" in kinds


def test_extract_no_quantities_in_pure_prose():
    text = "The Contractor shall maintain reasonable levels of workmanship."
    qs = extract_quantities([_clause(text)])
    # Should be empty, nothing matched
    assert len(qs) == 0


def test_extract_records_quantity_id_and_clause_id():
    qs = extract_quantities([_clause("365 days", clause_id="X42")])
    assert qs[0].clause_id == "X42"
    assert qs[0].quantity_id.startswith("X42::q")


def test_extract_captures_surrounding_text():
    text = "The Contractor shall complete the Works within 365 days of LoA issuance."
    qs = extract_quantities([_clause(text)])
    assert "365" in qs[0].surrounding_text
    assert qs[0].surrounding_text != qs[0].raw_text  # has more context


# ─── Unit normalisation ────────────────────────────────────────────────────
@pytest.mark.parametrize("value,unit,kind,expected", [
    (180, "days", "duration", (180, "days")),
    (6, "months", "duration", (180, "days")),
    (1, "year", "duration", (365, "days")),
    (50, "lakh", "money", (50, "inr_lakhs")),
    (5, "crore", "money", (500, "inr_lakhs")),
    (1, "m", "dimension", (1000, "mm")),
    (1, "cm", "dimension", (10, "mm")),
    (75, "mm", "dimension", (75, "mm")),
    (5, "%", "percentage", (5, "pct")),
])
def test_normalise(value, unit, kind, expected):
    assert _normalise(value, unit, kind) == expected
