"""Synthetic tender generator.

Generates a fake but realistic Indian-public-works tender package on disk so
later pipeline stages (parser, chunker, BCT, retrieval, rewrite) can be built
and tested without needing real CPPP data.

The synthetic tender:
  * Has 5 PDFs: NIT, GCC, SCC, Specs, BoQ
  * Has 2 corrigenda
  * Includes deliberately-vague clauses (target Type F) for BCT testing
  * Includes deliberately-precise clauses for negative examples
  * Includes one cross-document numerical inconsistency (180 days vs 6 months)
    even though numerical inconsistency is out of scope for v1 — useful for
    evaluating false-positive rate of the BCT
  * Each corrigendum amends a known vague clause so silver labels are
    derivable end-to-end

Usage:
    python -m src.acquire synthesize --tender-id SYN_001 --out corpus/

Or programmatically:
    from src.acquire.synthetic import generate_synthetic_tender
    folder = generate_synthetic_tender(out_root, tender_id="SYN_001")
"""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from src.acquire.manifest import (
    Manifest,
    TenderDocument,
    sha256_of_file,
    write_manifest,
)


# ─── Clauses with embedded ground-truth labels ──────────────────────────────
# Each clause is tagged with whether it should fire BCT (vague=True) and
# whether its v1 form was later corrected by a corrigendum.

NIT_TEXT = """\
NOTICE INVITING TENDER

Tender ID: SYN-2026-001
Issuing Authority: National Highways Authority of India (Synthetic Demo Authority)

1.1  Scope. This tender invites sealed bids from eligible contractors for
the construction of a 4-lane access road in compliance with MoRTH 5.0
Specifications.

1.2  Estimated cost. The estimated cost of works is INR 250 crore.

1.3  Bid submission. Bids shall be submitted electronically through the CPP
Portal not later than 31 March 2026, 1500 hours IST.

1.4  Pre-bid meeting. A pre-bid meeting shall be held on a suitable date,
which will be notified to bidders separately.

1.5  Eligibility. Bidders shall demonstrate adequate financial capacity and
relevant past experience. Specific thresholds are given in the Special
Conditions of Contract, Clause 3.
"""

GCC_TEXT = """\
GENERAL CONDITIONS OF CONTRACT

4.1  Definitions. In these Conditions, "the Contractor" means the entity
whose tender has been accepted by the Authority and with whom the Contract
has been entered into. "the Engineer" means the person appointed by the
Authority to administer the Contract under Sub-Clause 4.6 hereof.

4.2  Commencement. The Contractor shall commence the Works within 15 days
of the issue of the Letter of Acceptance and shall complete the Works
within 365 days of commencement.

4.3  Workmanship. The Contractor shall execute the Works in a workmanlike
manner using best industry practices and to the satisfaction of the
Engineer.

4.4  Materials. All materials shall be of suitable quality and procured
from approved sources. The Engineer reserves the right to reject any
material that does not conform to the relevant Indian Standard.

4.5  Defect Liability Period. The Defect Liability Period shall be 12
months from the date of issue of the Completion Certificate.

4.6  Variations. The Engineer may, by notice in writing, order any variation
to the Works as may be reasonably necessary. The Contractor shall be
entitled to fair compensation for any such variation, the value of which
shall be determined by the Engineer.

4.7  Termination. The Authority may terminate this Contract for cause as
specified in Clause 18, or for convenience upon 30 days' notice.
"""

SCC_TEXT = """\
SPECIAL CONDITIONS OF CONTRACT

5.1  Performance security. The Contractor shall furnish a performance
security in a form acceptable to the Authority, equal to 5% of the
Contract Price, valid until 90 days after the end of the Defect Liability
Period.

5.2  Insurance. The Contractor shall procure and maintain adequate
insurance covering all risks associated with the Works for the full
duration of the contract.

5.3  Environmental compliance. The Contractor shall maintain reasonable
levels of dust suppression during demolition and earthwork activities, and
shall take appropriate measures to mitigate noise and vibration impact on
neighbouring properties.

5.4  Safety. The Contractor shall provide a sufficient number of qualified
safety officers at the Site at all times during execution. The Contractor
shall ensure that all workers are appropriately trained.

5.5  Reporting. The Contractor shall submit progress reports to the
Engineer at regular intervals throughout the contract period.

5.6  Quality control. Quality control testing shall be carried out as per
the relevant IS code with sufficient frequency to ensure compliance.

5.7  Local employment. The Contractor shall make best efforts to employ
local labour wherever feasible.
"""

SPECS_TEXT = """\
TECHNICAL SPECIFICATIONS

6.1  Concrete. All structural concrete shall be of grade M30, manufactured
and placed in accordance with IS 456:2000 Section 9. Cement shall conform
to IS 269:2015 (Ordinary Portland Cement, 53 grade). Aggregate shall
conform to IS 383:2016. Slump for placed concrete shall be 75 ± 25 mm
measured at the point of placement.

6.2  Reinforcement steel. Reinforcement bars shall be high-strength
deformed bars conforming to IS 1786:2008, grade Fe 500D, of nominal
diameters as shown on the drawings.

6.3  Compaction. Subgrade compaction shall achieve a minimum dry density
of 98% of MDD as per IS 2720 Part 7, measured by sand replacement method
at intervals not exceeding 200 m of completed road length.

6.4  Bitumen. Bituminous concrete wearing course shall be 50 mm thick,
laid at a temperature not less than 145 °C, complying with MoRTH Section
509.

6.5  Drainage. Side drains shall be of trapezoidal cross-section with bed
width 0.5 m, side slopes 1:1.5, and minimum longitudinal gradient of 0.3%.

6.6  Curing. Concrete shall be cured by ponding or wet hessian for a
minimum of 7 days from placement, kept continuously moist throughout the
curing period.

6.7  Aesthetic finish. The finished surface shall present an aesthetically
pleasing appearance and be free from visible defects to the extent
practicable.
"""

BOQ_TEXT = """\
BILL OF QUANTITIES (Preamble + selected items)

Preamble Note 1. Rates entered in this BoQ shall be deemed to include all
labour, materials, plant, supervision, profits, overheads, and all other
costs reasonably incurred or to be incurred by the Contractor in execution
of the items, whether expressly mentioned or not.

Preamble Note 2. Items shall be measured net as built. No allowance shall
be made for waste, breakage, or shrinkage unless specifically stated.

Preamble Note 7. Items in this BoQ have been priced on the basis of
project completion within 6 months of the date of the LoA, as required
by Special Conditions Clause 5.5 reporting cycle. Any extension of this
period shall entitle the Contractor to a price-adjustment claim.

Item 1.01  Excavation in all types of soil including disposal within
500 m. Unit: cum. Quantity: 12,500.

Item 2.01  Plain cement concrete grade M15, including formwork. Unit:
cum. Quantity: 850.

Item 2.02  Reinforced cement concrete grade M30, including formwork,
reinforcement extra. Unit: cum. Quantity: 2,400.

Item 2.03  High-yield-strength deformed reinforcement bars (Fe 500D)
including cutting, bending, placing. Unit: tonne. Quantity: 320.

Item 4.01  Bituminous concrete wearing course, 50 mm thick. Unit: sqm.
Quantity: 18,000.
"""

# A corrigendum that amends Clause 5.3 to be precise (PM10 limit specified).
CORRIGENDUM_001_TEXT = """\
CORRIGENDUM NO. 1

Issued: 12 February 2026
Tender ID: SYN-2026-001

The following amendment is hereby made to the tender document published
on 28 January 2026:

1. In Special Conditions of Contract, Clause 5.3 (Environmental
compliance), the words "shall maintain reasonable levels of dust
suppression during demolition and earthwork activities, and shall take
appropriate measures to mitigate noise and vibration impact on neighbouring
properties" are hereby replaced with:

  "shall maintain dust suppression such that PM10 concentrations measured
  at the site boundary do not exceed 100 µg/m³ over a 24-hour averaging
  period, in accordance with the Central Pollution Control Board's
  ambient air quality standards. Noise levels at the site boundary shall
  not exceed 75 dB(A) during the day (06:00-22:00) and 70 dB(A) during
  the night, as per IS 9989."

All other terms and conditions of the tender remain unchanged.
"""

# A corrigendum that amends Clause 5.5 to be precise (interval specified).
CORRIGENDUM_002_TEXT = """\
CORRIGENDUM NO. 2

Issued: 24 February 2026
Tender ID: SYN-2026-001

The following amendment is hereby made to the tender document published
on 28 January 2026, as already amended by Corrigendum No. 1:

1. In Special Conditions of Contract, Clause 5.5 (Reporting), the words
"at regular intervals throughout the contract period" are hereby replaced
with:

  "weekly, in the format prescribed in Annex C of these Special Conditions,
  delivered to the Engineer in both hard copy and electronic form not
  later than 1500 hours each Monday for the preceding seven-day period."

2. In Bill of Quantities, Preamble Note 7, the words "within 6 months of
the date of the LoA" are hereby replaced with:

  "within 365 days of commencement, consistent with General Conditions of
  Contract Clause 4.2."

All other terms and conditions of the tender remain unchanged.
"""


# ─── PDF generation ─────────────────────────────────────────────────────────
def _text_to_pdf(text: str, out_path: Path, title: str) -> Path:
    """Render plain text to a multi-page PDF using PyMuPDF.

    Each page is A4. We use a fixed-width-ish 11pt font and wrap at ~85
    characters per line. Pagination is automatic.
    """
    import fitz  # PyMuPDF

    doc = fitz.open()
    page_w, page_h = fitz.paper_size("a4")  # ~595 x 842 points
    margin = 54  # 0.75 inch
    line_h = 14
    font_size = 11
    font_name = "helv"

    title_lines = title.splitlines() or [title]
    body_lines: list[str] = []
    for raw in text.splitlines():
        if not raw:
            body_lines.append("")
            continue
        # Soft-wrap at 88 chars
        while len(raw) > 88:
            cut = raw.rfind(" ", 0, 88)
            if cut == -1:
                cut = 88
            body_lines.append(raw[:cut])
            raw = raw[cut:].lstrip()
        body_lines.append(raw)

    # Now render with simple pagination
    page = doc.new_page(width=page_w, height=page_h)
    cursor_y = margin

    # Title block on first page
    page.insert_text(
        (margin, cursor_y),
        title,
        fontsize=14,
        fontname="helv",
    )
    cursor_y += 28

    for line in body_lines:
        if cursor_y > page_h - margin:
            page = doc.new_page(width=page_w, height=page_h)
            cursor_y = margin
        page.insert_text(
            (margin, cursor_y),
            line,
            fontsize=font_size,
            fontname=font_name,
        )
        cursor_y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    doc.close()
    return out_path


# ─── Top-level generator ────────────────────────────────────────────────────
def generate_synthetic_tender(
    out_root: Path,
    *,
    tender_id: str = "SYN_001",
    overwrite: bool = False,
) -> Path:
    """Generate a complete synthetic tender folder. Returns the folder path.

    The folder will contain: NIT, GCC, SCC, Specs, BoQ, two corrigenda, and
    a manifest.json that ties them all together.
    """
    folder = Path(out_root) / tender_id
    if folder.exists() and not overwrite:
        if any(folder.iterdir()):
            raise FileExistsError(
                f"{folder} already exists and is not empty. "
                f"Pass overwrite=True to clobber."
            )
    folder.mkdir(parents=True, exist_ok=True)

    files_to_write: list[tuple[str, str, str]] = [
        # (filename, title, body)
        ("01_nit.pdf",            "Notice Inviting Tender — SYN-2026-001",          NIT_TEXT),
        ("02_gcc.pdf",            "General Conditions of Contract — SYN-2026-001",  GCC_TEXT),
        ("03_scc.pdf",            "Special Conditions of Contract — SYN-2026-001",  SCC_TEXT),
        ("04_specs.pdf",          "Technical Specifications — SYN-2026-001",        SPECS_TEXT),
        ("05_boq.pdf",            "Bill of Quantities — SYN-2026-001",              BOQ_TEXT),
        ("corrigendum_001.pdf",   "Corrigendum No. 1 — SYN-2026-001",               CORRIGENDUM_001_TEXT),
        ("corrigendum_002.pdf",   "Corrigendum No. 2 — SYN-2026-001",               CORRIGENDUM_002_TEXT),
    ]

    documents: list[TenderDocument] = []

    for filename, title, body in files_to_write:
        out_path = folder / filename
        _text_to_pdf(body, out_path, title)

        # Build the manifest entry
        from src.acquire.pdf_validator import inspect_pdf

        info = inspect_pdf(out_path, sample_text_layer=True)
        if filename.startswith("corrigendum_"):
            from src.acquire.manifest import infer_corrigendum_sequence

            role = "corrigendum"
            seq = infer_corrigendum_sequence(filename)
            issuance_date: Optional[date] = (
                date(2026, 2, 12) if seq == 1 else date(2026, 2, 24)
            )
        else:
            from src.acquire.manifest import infer_role_from_filename

            role = infer_role_from_filename(filename)
            seq = 0
            issuance_date = None

        documents.append(
            TenderDocument(
                filename=filename,
                role=role,
                sequence=seq,
                page_count=info.page_count,
                file_size_bytes=info.file_size_bytes,
                sha256=sha256_of_file(out_path),
                issuance_date=issuance_date,
                notes="" if role != "other" else "auto-tagged 'other'",
            )
        )

    manifest = Manifest(
        tender_id=tender_id,
        source_url="(synthetic — generated locally for pipeline development)",
        issuing_authority="Synthetic Demo Authority (for development only)",
        project_title="Synthetic 4-lane access road, demo tender",
        issue_date=date(2026, 1, 28),
        award_date=None,
        project_value_inr_lakhs=25000.0,  # 250 crore
        documents=documents,
        retrieved_at=datetime.now(tz=timezone.utc),
        retrieved_by="generate_synthetic_tender",
        notes=(
            "Synthetic tender for development. NOT a real CPPP tender. "
            "Includes deliberately vague clauses and two corrigenda that "
            "resolve them, suitable for end-to-end pipeline testing."
        ),
    )

    write_manifest(folder, manifest)
    return folder
