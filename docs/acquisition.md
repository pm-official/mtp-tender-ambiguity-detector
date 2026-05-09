# Tender + corrigendum acquisition

This document describes the three workflows for getting a tender package
into `corpus/<tender_id>/` so the rest of the pipeline can process it.

## TL;DR — choose your workflow

| Workflow | When to use | What you do |
|---|---|---|
| **A. Synthetic** | Building or debugging the pipeline; no real tender on hand | One CLI command. PDFs and manifest are generated end-to-end. |
| **B. Drop folder** | You manually downloaded PDFs from CPPP / NHAI / CPWD | Drop PDFs into a folder, run one CLI command, manifest is built. |
| **C. Selenium scrape** | You want a guided, semi-automatic flow with the Chrome browser | Run a CLI command, solve any CAPTCHAs in the browser, scraper takes over. |

All three workflows produce the same output: `corpus/<tender_id>/` with PDFs
plus a `manifest.json` that downstream stages trust.

---

## Workflow A — Synthetic tender (for development)

The synthetic generator creates a realistic Indian-public-works tender
package on disk. It includes 5 base PDFs (NIT, GCC, SCC, Specs, BoQ) plus
2 corrigenda. The corrigenda deliberately amend two vague clauses in the
SCC, so the silver-label generator (Session 5) and the BCT (Session 6) can
both be exercised end-to-end without any real tender data.

### Run it

```bash
python -m src.cli acquire synthesize --tender-id SYN_001
```

You can generate as many as you want with different IDs:

```bash
python -m src.cli acquire synthesize --tender-id SYN_002
python -m src.cli acquire synthesize --tender-id SYN_003
```

### What you get

```
corpus/SYN_001/
├── 01_nit.pdf
├── 02_gcc.pdf
├── 03_scc.pdf            <- contains 2 deliberately vague clauses
├── 04_specs.pdf
├── 05_boq.pdf
├── corrigendum_001.pdf   <- amends Clause 5.3 (PM10 limit added)
├── corrigendum_002.pdf   <- amends Clauses 5.5 and BoQ Note 7
└── manifest.json
```

Each PDF is a real, valid PDF (PyMuPDF-generated). Page counts and SHA256s
populate the manifest automatically.

### What it's good for

- Verifying a parser produces sensible chunks before a real tender lands.
- Testing the BCT prompt against known-vague clauses ("reasonable levels of
  dust suppression", "sufficient number of qualified safety officers", etc.).
- Testing the corrigendum-diff silver-label generator (Session 5) — every
  vague clause has a known v2 from a corrigendum.

### What it's NOT for

- Reporting research metrics in the thesis. The synthetic tender is for
  development only; final results must use real CPPP tenders.

---

## Workflow B — Drop folder (the recommended path for real tenders)

This is the realistic workflow given that CPPP gates every search behind a
CAPTCHA. You do the discovery and download manually; the tooling does the
manifest-building.

### Step 1 — Find a tender on CPPP

1. Open https://eprocure.gov.in/eprocure/app
2. Click **Tender Status** in the top navigation.
3. Filter for:
   - `Tender Status = AOC` (Award of Contract issued — i.e. closed tenders)
   - `Tender Category = Works`
   - Optionally `Form of Contract = EPC Contract` for big NHAI tenders
4. Solve the CAPTCHA, click **Search**.
5. Pick a tender that has at least 3 corrigenda issued (visible in the
   tender detail page). Project value above ₹50 crore is a good filter
   for high-quality drafting.

Good candidates for the corpus:

- 3-4 NHAI EPC tenders
- 1-2 NHAI HAM tenders
- 2-3 CPWD building works tenders
- 1-2 BMC tenders
- 1-2 state PWD (Maharashtra / Delhi)

### Step 2 — Download all PDFs into the tender folder

1. Decide a tender ID for yourself, e.g. `NHAI_2024_DEL_MUM_PKG3`.
2. Create the folder via CLI:
   ```bash
   python -m src.cli acquire init \
       --tender-id NHAI_2024_DEL_MUM_PKG3 \
       --authority "National Highways Authority of India" \
       --title "Delhi-Mumbai Expressway Package 3" \
       --source-url "https://etenders.gov.in/.../tenderId=..."
   ```
3. From the CPPP tender detail page, download EVERY linked PDF:
   - NIT (Notice Inviting Tender)
   - GCC (General Conditions of Contract)
   - SCC / Particular Conditions / Additional Conditions
   - Technical Specifications (may be one file or many)
   - BoQ (Bill of Quantities)
   - Drawings list / index (if available; actual drawings are often too large)
   - Annexures / Appendices / Schedules
   - **Every Corrigendum** — these are critical for silver-label generation
4. Save them all into `corpus/NHAI_2024_DEL_MUM_PKG3/`.
5. (Optional) Rename them so role inference works — files with `corrigendum`
   in the name are auto-tagged. See the role list at the top of
   `src/acquire/manifest.py`.

### Step 3 — Build the manifest

```bash
python -m src.cli acquire drop --tender-id NHAI_2024_DEL_MUM_PKG3
```

This:
- Inspects every PDF in the folder.
- Validates each is a real PDF and not a download error page.
- Auto-tags roles from filenames (corrigendum, NIT, GCC, SCC, etc.).
- Captures page counts, file sizes, SHA256 hashes.
- Writes `manifest.json`.

### Step 4 — Verify

```bash
python -m src.cli acquire validate --tender-id NHAI_2024_DEL_MUM_PKG3
```

Reports OK or lists any issues. Common issues:
- A PDF on disk is missing from the manifest (re-run `drop`).
- A PDF in the manifest is missing from disk (you deleted or moved it).
- SHA256 mismatch (you replaced a file since the last `drop`).

### Step 5 — Inspect the manifest, fix wrong roles

Open `corpus/<tender_id>/manifest.json` in a text editor. If a file was
auto-tagged `other` but is actually e.g. a `boq`, change the `role` field.
Re-running `validate` afterwards is fine — your edits are preserved unless
you re-run `drop`.

### List all tenders in the corpus

```bash
python -m src.cli acquire list
```

Quick summary table of every tender folder.

---

## Workflow C — Selenium semi-automatic scrape (optional)

For repeated downloads when you don't want to click "Save" on every PDF.

### Caveats

- CPPP uses CAPTCHAs that the scraper cannot solve. You navigate manually
  to the tender detail page in the Chrome window the scraper opens.
- Some CPPP downloads are POST forms; the scraper handles common cases
  but a few might still need a manual click.
- Not every tender will scrape cleanly — fall back to Workflow B if so.

### Run it

```bash
python -m src.cli acquire scrape --tender-id NHAI_2024_PKG3
```

The CLI will:
1. Open a fresh Chrome window via Selenium.
2. Configure the browser to auto-download all PDFs into the tender folder.
3. Start at https://etenders.gov.in/eprocure/app.
4. Wait for you to navigate manually to the tender detail page.
5. When you press Enter in the terminal, find every PDF link on the
   current page and trigger downloads.
6. Wait for downloads to complete.
7. Build the manifest from whatever ended up in the folder.

After the scrape, run `validate` to confirm the result looks right.

### When this saves time

Roughly when downloading 20+ files from a single tender — the scraper
fires them all in parallel rather than you clicking through each one.

For 5-7 file tenders, Workflow B is faster.

---

## Selection criteria for the 8-12 tender corpus

Per the methodology document Part 4.1, the target corpus is 8-12 already-
awarded Indian public-works tenders matching:

- **Project value** above ₹50 crore (richer drafting, more bidder scrutiny → more corrigenda)
- **Document length** 200-800 pages total (large enough to be realistic, small enough to process during development)
- **At least 3 corrigenda** issued during the bid window (silver-label material)
- **English only** (Hindi / Hindi-mixed is out of scope for v1)
- **AOC issued** (corrigendum stream is complete)

Recommended distribution:

| Authority | Target count |
|---|---|
| NHAI EPC | 3-4 |
| NHAI HAM | 1-2 |
| CPWD building works | 2-3 |
| BMC | 1-2 |
| State PWD (Maharashtra / Delhi) | 1-2 |

---

## File-naming convention (recommended, not mandatory)

The role-inference heuristic looks for these substrings in filenames. If
your downloads already use these words, the manifest auto-tags correctly.

| Substring (case-insensitive) | Auto-tagged role |
|---|---|
| `corrigendum`, `corr_`, `addendum`, `addenda`, `amend` | `corrigendum` |
| `nit`, `notice_inviting`, `invitation` | `nit` |
| `gcc`, `general_conditions` | `gcc` |
| `scc`, `special_conditions`, `particular_conditions`, `additional_conditions` | `scc` |
| `specifications`, `technical_specs`, `specs` | `specs` |
| `boq`, `bill_of_quantities`, `schedule_of_rates`, `schedule_of_quantities` | `boq` |
| `drawings`, `drawing_list` | `drawings` |
| `annex`, `appendix`, `schedule` | `annex` |
| `tender` (only if nothing else matched) | `tender_v1` |
| anything else | `other` (review and edit manually) |

For corrigenda, the auto-tagger also pulls a sequence number from the
filename: `corrigendum_001.pdf` → sequence 1, `addendum_3.pdf` → sequence 3.

---

## What to do next

After you have at least one tender in `corpus/`, move to Session 3 (PDF
parser + clause-aware chunker). The parser reads from `corpus/<tender_id>/`,
honours the manifest, and produces `clauses.jsonl` per tender.

Session 5 (corrigendum diff + silver-label generator) is what consumes the
corrigendum PDFs. If your tenders have only 0-1 corrigenda, silver-label
recall will be weak — pick tenders with richer correction streams when you
can.
