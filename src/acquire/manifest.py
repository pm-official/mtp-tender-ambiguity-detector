"""Tender manifest schema and I/O helpers.

A `manifest.json` lives in every `corpus/<tender_id>/` directory and records
what was downloaded, when, from where, and how to interpret each PDF in the
folder. The manifest is the source of truth that downstream stages (parser,
silver-label generator, evaluation) trust.

Roles used for tender PDFs (kept short so they're easy to type at the CLI):

  tender_v1   — the original tender package master file (NIT bound with GCC)
  nit         — Notice Inviting Tender (standalone)
  gcc         — General Conditions of Contract (standalone)
  scc         — Special / Particular / Additional Conditions
  specs       — Technical Specifications
  boq         — Bill of Quantities
  drawings    — Drawings index or drawings volume
  annex       — Annexures or schedules
  corrigendum — A corrigendum / addendum (sequence number = 1, 2, 3, ...)
  other       — Anything else (audit, AOC, pre-bid query register, etc.)
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_serializer

# All valid roles. Keep this list small and stable.
DocumentRole = Literal[
    "tender_v1",
    "nit",
    "gcc",
    "scc",
    "specs",
    "boq",
    "drawings",
    "annex",
    "corrigendum",
    "other",
]

VALID_ROLES = {
    "tender_v1", "nit", "gcc", "scc", "specs", "boq",
    "drawings", "annex", "corrigendum", "other",
}


class TenderDocument(BaseModel):
    """One PDF inside a tender folder."""

    filename: str
    role: DocumentRole
    sequence: int = 0   # 1, 2, 3 for corrigenda; 0 otherwise
    page_count: int = 0
    file_size_bytes: int = 0
    sha256: str = ""
    issuance_date: Optional[date] = None
    notes: str = ""

    @field_serializer("issuance_date")
    def _ser_date(self, v: Optional[date]) -> Optional[str]:
        return v.isoformat() if v else None


class Manifest(BaseModel):
    """Per-tender provenance record."""

    tender_id: str
    source_url: str = ""
    issuing_authority: str = "UNKNOWN"
    project_title: str = ""
    issue_date: Optional[date] = None
    award_date: Optional[date] = None
    project_value_inr_lakhs: Optional[float] = None

    documents: list[TenderDocument] = Field(default_factory=list)

    retrieved_at: datetime
    retrieved_by: str = "Prakhar"
    notes: str = ""
    schema_version: str = "1.0"

    # ── Convenience helpers ──────────────────────────────────────────────
    @property
    def num_corrigenda(self) -> int:
        return sum(1 for d in self.documents if d.role == "corrigendum")

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    def corrigenda_in_order(self) -> list[TenderDocument]:
        return sorted(
            (d for d in self.documents if d.role == "corrigendum"),
            key=lambda d: d.sequence,
        )

    def document_by_filename(self, filename: str) -> Optional[TenderDocument]:
        for d in self.documents:
            if d.filename == filename:
                return d
        return None

    @field_serializer("issue_date", "award_date")
    def _ser_date(self, v: Optional[date]) -> Optional[str]:
        return v.isoformat() if v else None

    @field_serializer("retrieved_at")
    def _ser_datetime(self, v: datetime) -> str:
        return v.isoformat()


# ─── I/O ────────────────────────────────────────────────────────────────────
def manifest_path(tender_dir: Path) -> Path:
    return Path(tender_dir) / "manifest.json"


def write_manifest(tender_dir: Path, manifest: Manifest) -> Path:
    """Write manifest.json into tender_dir. Returns the path written."""
    out = manifest_path(tender_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json", exclude_none=False)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out


def read_manifest(tender_dir: Path) -> Manifest:
    """Read and validate manifest.json from tender_dir."""
    p = manifest_path(tender_dir)
    if not p.exists():
        raise FileNotFoundError(f"No manifest.json in {tender_dir}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    try:
        return Manifest.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid manifest at {p}:\n{e}") from e


def empty_manifest(tender_id: str) -> Manifest:
    """Construct an empty manifest with sensible defaults and current time."""
    return Manifest(
        tender_id=tender_id,
        retrieved_at=datetime.now(tz=timezone.utc),
    )


# ─── Filename → role inference ──────────────────────────────────────────────
# Lightweight heuristic for auto-tagging. The user can override via interactive
# prompts. Patterns are case-insensitive substring matches against the
# filename (stem), checked top-to-bottom — first match wins.
_FILENAME_ROLE_HINTS: list[tuple[str, DocumentRole]] = [
    ("corrigendum", "corrigendum"),
    ("corr_", "corrigendum"),
    ("addendum", "corrigendum"),
    ("addenda", "corrigendum"),
    ("amend", "corrigendum"),

    ("nit", "nit"),
    ("notice_inviting", "nit"),
    ("invitation", "nit"),

    ("gcc", "gcc"),
    ("general_conditions", "gcc"),

    ("scc", "scc"),
    ("special_conditions", "scc"),
    ("particular_conditions", "scc"),
    ("additional_conditions", "scc"),

    ("specifications", "specs"),
    ("technical_specs", "specs"),
    ("specs", "specs"),

    ("boq", "boq"),
    ("bill_of_quantities", "boq"),
    ("schedule_of_rates", "boq"),
    ("schedule_of_quantities", "boq"),

    ("drawings", "drawings"),
    ("drawing_list", "drawings"),

    ("annex", "annex"),
    ("appendix", "annex"),
    ("schedule", "annex"),

    ("tender", "tender_v1"),
]


def infer_role_from_filename(filename: str) -> DocumentRole:
    """Best-effort guess at document role based on filename. Returns 'other'
    if nothing matches."""
    stem = Path(filename).stem.lower().replace("-", "_").replace(" ", "_")
    for needle, role in _FILENAME_ROLE_HINTS:
        if needle in stem:
            return role
    return "other"


def infer_corrigendum_sequence(filename: str) -> int:
    """Pull a sequence number out of a corrigendum filename. Looks for
    patterns like corrigendum_001, corrigendum-2, corr3, etc. Returns 0 if
    not detected."""
    import re

    stem = Path(filename).stem.lower()
    # match the LAST run of digits in the stem (handles "corr_2024_001" etc.)
    matches = re.findall(r"\d+", stem)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            return 0
    return 0


# ─── SHA256 helper ──────────────────────────────────────────────────────────
def sha256_of_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
