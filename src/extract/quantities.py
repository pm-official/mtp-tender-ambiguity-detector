"""Numerical-quantity extractor.

Two-stage pipeline:
  Stage A — REGEX. Find every value-with-unit candidate. Cheap, deterministic.
  Stage B — LLM (batched, optional). Classify each candidate's "referent" —
            what does this number measure? ("defect liability period",
            "performance security percent", "concrete cube test frequency", ...)

Stage A output alone is often enough for downstream stages that only need
"give me all the durations in this tender" — so we make the LLM step opt-in.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field

from src.schemas.clause import Clause
from src.schemas.structures import Quantity

logger = logging.getLogger(__name__)


# ─── Regex patterns ─────────────────────────────────────────────────────────
# All patterns capture group 1 = value, group 2 = unit-token.
# Order matters; first match wins for overlapping patterns.

_QUANTITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Money — "Rs. 50 lakh", "INR 2.5 crore", "₹100 crores"
    (re.compile(r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)\s*(lakhs?|crores?|lakh|crore)?", re.IGNORECASE), "money"),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*(lakhs?|crores?|cr|lacs?)\b", re.IGNORECASE), "money"),
    (re.compile(r"USD\s*([\d,]+(?:\.\d+)?)\s*(million|m|billion|bn)?", re.IGNORECASE), "money"),
    # Tolerance — "± 2 mm", "+/- 5%"
    (re.compile(r"(?:±|\+/-|\+\s*-)\s*([\d.]+)\s*(mm|cm|m|%|degC|deg\s*C|kg)?", re.IGNORECASE), "tolerance"),
    # Percentage — "15%", "5 percent", "5 per cent". No trailing \b because
    # "%" is non-word; \b would fail when followed by whitespace.
    (re.compile(r"([\d.]+)\s*(?:%|per\s*cent\b|percent\b)", re.IGNORECASE), "percentage"),
    # Duration — "180 days", "6 months", "3 years"
    (re.compile(r"\b([\d.]+)\s*(days?|wks?|weeks?|mos?|months?|yrs?|years?|hours?|hrs?|minutes?|mins?)\b", re.IGNORECASE), "duration"),
    # Dimension — "450 mm", "2.5 m", "10 cm"
    (re.compile(r"\b([\d.]+)\s*(mm|cm|m|ft|inch(?:es)?|in)\b", re.IGNORECASE), "dimension"),
    # Frequency — "twice a week", "once a month" — coarse
    (re.compile(r"\b([\d]+)\s*(?:times?\s+)?(?:per|/)\s*(day|week|month|year)\b", re.IGNORECASE), "frequency"),
    # Ratio — "1:5", "2:3"
    (re.compile(r"\b([\d]+)\s*:\s*([\d]+)\b"), "ratio"),
    # Generic counts attached to common concepts. Keep last; otherwise too noisy.
    # Skipped for now to control false-positive rate.
]


# ─── Unit normalisation ─────────────────────────────────────────────────────
def _normalise(value: float, unit: str, kind: str) -> tuple[float, str]:
    """Convert (value, raw unit) to (canonical value, canonical unit)."""
    u = (unit or "").lower().strip()

    if kind == "duration":
        if u.startswith("hour") or u in {"hr", "hrs"}:
            return value / 24, "days"
        if u.startswith("min"):
            return value / (24 * 60), "days"
        if u.startswith("day") or u in {"d"}:
            return value, "days"
        if u.startswith("wk") or u.startswith("week"):
            return value * 7, "days"
        if u.startswith("mo") or u in {"mos"}:
            return value * 30, "days"
        if u.startswith("yr") or u.startswith("year"):
            return value * 365, "days"
        return value, "days"

    if kind == "money":
        if "crore" in u or u.startswith("cr"):
            return value * 100, "inr_lakhs"
        if "lakh" in u or u.startswith("lac"):
            return value, "inr_lakhs"
        if "million" in u or u in {"m"}:
            return value * 84.0, "inr_lakhs"   # USD -> INR rough convert
        if "billion" in u or u in {"bn"}:
            return value * 84_000.0, "inr_lakhs"
        return value / 100_000, "inr_lakhs"      # bare INR -> lakhs

    if kind == "dimension":
        if u == "mm":
            return value, "mm"
        if u == "cm":
            return value * 10, "mm"
        if u == "m":
            return value * 1000, "mm"
        if u in {"ft"}:
            return value * 304.8, "mm"
        if u.startswith("inch") or u == "in":
            return value * 25.4, "mm"
        return value, "mm"

    if kind == "percentage":
        return value, "pct"

    if kind == "tolerance":
        return value, u or "tolerance"

    if kind == "frequency":
        # canonical: per day equivalent
        if u == "day":
            return value, "per_day"
        if u == "week":
            return value / 7, "per_day"
        if u == "month":
            return value / 30, "per_day"
        if u == "year":
            return value / 365, "per_day"
        return value, "per_day"

    if kind == "ratio":
        return value, "ratio"

    return value, u or "raw"


# ─── Surrounding-context helper ─────────────────────────────────────────────
def _surrounding_window(clause_text: str, span_start: int, span_end: int,
                        chars_each_side: int = 80) -> str:
    s = max(0, span_start - chars_each_side)
    e = min(len(clause_text), span_end + chars_each_side)
    return re.sub(r"\s+", " ", clause_text[s:e]).strip()


# ─── Top-level extractor ────────────────────────────────────────────────────
def extract_quantities(clauses: Iterable[Clause]) -> list[Quantity]:
    """Run regex pass and return Quantity objects (without referent labels)."""
    out: list[Quantity] = []
    for clause in clauses:
        seen_spans: list[tuple[int, int]] = []  # avoid double-counting nested matches
        local_idx = 0
        for pattern, kind in _QUANTITY_PATTERNS:
            for m in pattern.finditer(clause.text):
                start, end = m.span()
                # skip spans that overlap a previously-recorded one
                if any(not (end <= s or start >= e) for s, e in seen_spans):
                    continue
                seen_spans.append((start, end))

                raw = m.group(0).strip()
                try:
                    if kind == "ratio":
                        val_a = float(m.group(1).replace(",", ""))
                        val_b = float(m.group(2).replace(",", "")) if m.lastindex and m.lastindex >= 2 else 0
                        value = val_a / val_b if val_b else val_a
                        unit = "ratio"
                    else:
                        value = float(m.group(1).replace(",", ""))
                        unit = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
                except (TypeError, ValueError):
                    continue

                value_norm, unit_norm = _normalise(value, unit or "", kind)
                surround = _surrounding_window(clause.text, start, end)
                out.append(Quantity(
                    quantity_id=f"{clause.clause_id}::q{local_idx:03d}",
                    clause_id=clause.clause_id,
                    doc_id=clause.doc_id,
                    tender_id=clause.tender_id,
                    raw_text=raw,
                    surrounding_text=surround,
                    kind=kind,
                    value=value,
                    unit=unit or "",
                    value_norm=value_norm,
                    unit_norm=unit_norm,
                    page=clause.page,
                ))
                local_idx += 1
    return out


def classify_quantity_referents(
    quantities: list[Quantity],
    *,
    batch_size: int = 30,
    vocab_path: Optional[Path] = None,
) -> list[Quantity]:
    """Send each quantity to Gemini Flash with its surrounding text and ask
    for a canonical referent label. Mutates and returns the list."""
    if not quantities:
        return quantities

    from src.llm.gemini_client import get_default_client
    client = get_default_client()
    vocab = _load_vocab(vocab_path)

    class _Item(BaseModel):
        quantity_id: str
        referent: str
        confidence: float

    class _Reply(BaseModel):
        items: list[_Item] = Field(default_factory=list)

    for i in range(0, len(quantities), batch_size):
        batch = quantities[i : i + batch_size]
        prompt = _build_referent_prompt(batch, vocab)
        try:
            reply, _ = client.generate(
                prompt=prompt,
                response_schema=_Reply,
                model=client.default_flash,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("Referent classification batch failed (skipping): %s", e)
            continue

        by_id = {q.quantity_id: q for q in batch}
        for item in reply.items:
            q = by_id.get(item.quantity_id)
            if not q:
                continue
            q.referent = (item.referent or "UNCLASSIFIED").strip().lower().replace(" ", "_")
            q.referent_confidence = max(0.0, min(1.0, float(item.confidence)))
            q.referent_method = "llm"

    return quantities


def _load_vocab(path: Optional[Path]) -> list[str]:
    if path is None or not Path(path).exists():
        return _DEFAULT_REFERENT_VOCAB
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


_DEFAULT_REFERENT_VOCAB = [
    "defect_liability_period",
    "completion_period",
    "mobilisation_advance_pct",
    "mobilisation_advance_amount",
    "performance_security_pct",
    "liquidated_damages_pct",
    "liquidated_damages_cap",
    "earnest_money_deposit_amount",
    "tender_fee_amount",
    "contract_value_estimate",
    "validity_of_bid",
    "advance_payment_pct",
    "retention_money_pct",
    "price_adjustment_threshold",
    "concrete_grade",
    "steel_grade",
    "cube_test_frequency",
    "compaction_density_pct",
    "slump_value_mm",
    "tolerance_mm",
    "minimum_thickness_mm",
    "maximum_settlement_mm",
    "noise_limit_db",
    "particulate_matter_limit",
    "cement_quantity_per_cum",
    "boq_item_quantity",
    "completion_milestone_pct",
    "warranty_period",
    "notice_period",
    "interest_rate_pct",
    "vat_or_gst_rate_pct",
    "manpower_count",
    "tender_processing_fee",
    "minimum_turnover_amount",
]


def _build_referent_prompt(batch: list[Quantity], vocab: list[str]) -> str:
    items_block = []
    for q in batch:
        items_block.append(
            f"  [{q.quantity_id}] kind={q.kind} value={q.value} unit={q.unit!r}\n"
            f"     context: {q.surrounding_text[:240]}"
        )
    return (
        "You classify the canonical REFERENT of a numerical quantity found in "
        "an Indian construction tender. The referent is what the number "
        "MEASURES — for example a duration that is the defect-liability "
        "period, or a percentage that is the performance security.\n\n"
        "Use the controlled vocabulary below where possible. If none fits, "
        "propose a new short snake_case label.\n\n"
        f"Controlled vocabulary:\n  {', '.join(vocab)}\n\n"
        "Quantities:\n" + "\n".join(items_block) + "\n\n"
        "For EACH quantity, return strict JSON:\n"
        "{ 'items': [ {quantity_id, referent, confidence}, ... ] }\n"
        "Confidence is 0..1. Use lowercase snake_case for referent values."
    )


# ─── I/O ────────────────────────────────────────────────────────────────────
def write_quantities(out_path: Path, qs: list[Quantity]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(q.model_dump_json() + "\n")


def read_quantities(in_path: Path) -> list[Quantity]:
    in_path = Path(in_path)
    if not in_path.exists():
        return []
    out: list[Quantity] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Quantity.model_validate_json(line))
    return out
