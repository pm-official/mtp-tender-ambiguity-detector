# Setup

## Prerequisites

- Python 3.11 or newer
- Git
- ~2 GB free disk for the IS-code corpus (PDFs + ChromaDB index)
- A Gemini API key (free tier suffices for SYN_001; real-tender runs cost ₹500–₹1500)

## Install

```bash
# 1. Clone
git clone https://github.com/<your-username>/tender-ambiguity-detector.git
cd tender-ambiguity-detector

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell / CMD
# source .venv/bin/activate     # macOS / Linux

# 3. Editable install with dev extras
pip install -e ".[dev]"

# 4. Environment file
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

Open `.env` and replace `YOUR_GEMINI_API_KEY_HERE` with your key from https://aistudio.google.com/app/apikey.

## Verify install

```bash
# Offline tests — no network required
pytest tests/ -q

# Live smoke test — confirms your key works
pytest tests/test_gemini_smoke.py -v
```

You should see `246 passed` (offline) and `3 passed` (smoke). If the smoke test errors with `RESOURCE_EXHAUSTED`, your free-tier quota is hit; wait a minute and re-run.

## IS-code corpus (one-time, ~2 minutes, ~₹15)

The Stage 4 rewriter cites sections of Indian Standard codes. To verify those citations against ground truth, the system needs a local index of section-level chunks of each code.

### Source PDFs

Place the following PDFs into `resources/iscode/` (filenames matter — they encode the code id and version year used by the parser):

| Filename | What |
|---|---|
| `IS_269_2013_OPC_33Grade.pdf` | Ordinary Portland Cement, 33-grade |
| `IS_383_2016_AggregatesForConcrete.pdf` | Aggregates for Concrete |
| `IS_456_2000_PlainAndReinforcedConcrete.pdf` | Plain and Reinforced Concrete (Code of Practice) |
| `IS_800_2007_GeneralConstructionInSteel.pdf` | Steel construction |
| `IS_808_1989_HotRolledSteelSections.pdf` | Hot-rolled steel sections |
| `IS_875_part1_1987_DeadLoads.pdf` | Loads for buildings — dead loads |
| `IS_875_part2_1987_LiveLoads.pdf` | Imposed (live) loads |
| `IS_875_part3_2015_WindLoads.pdf` | Wind loads |
| `IS_875_part4_1987_SnowLoads.pdf` | Snow loads |
| `IS_875_part5_1987_LoadsForSpecialCases.pdf` | Special loads / load combinations |
| `IS_1200_part1_1992_GeneralProvisions.pdf` | Method of measurement — general |
| `IS_1200_part2_1974_Concrete.pdf` | Method of measurement — concrete |
| `IS_1200_part3_1976_BrickWork.pdf` | Brick work |
| `IS_1200_part5_1982_FormWork.pdf` | Formwork |
| `IS_1200_part6_1974_Refractory.pdf` | Refractory work |
| `IS_1200_part7_1972_HardwareAndIronmongery.pdf` | Hardware |
| `IS_1200_part8_1993_FlooringPavingAndFinishing.pdf` | Flooring |
| `IS_1200_part10_1973_Ceiling.pdf` | Ceilings |
| `IS_1200_part16_1979_WaterAndSewerLines.pdf` | Water and sewer lines |
| `IS_1200_part18_1973_DemolitionAndDismantling.pdf` | Demolition |
| `IS_1200_part19_1981_PaintingAndPolishing.pdf` | Painting |
| `IS_1200_part20_1981_LayingOfMetalRoadKerb.pdf` | Roadworks |
| `IS_1200_part23_1989_PilingAndOtherDeepFoundations.pdf` | Piling |
| `IS_1200_part25_1992_SteelStructures.pdf` | Steel structures |
| `IS_1786_2008_HSDSteelBars.pdf` | High-strength deformed bars |
| `IS_1893_part1_2002_EarthquakeResistantDesign.pdf` | Seismic design |
| `IS_13920_1993_DuctileDetailingOfReinforcedConcrete.pdf` | Ductile detailing |
| `CPWD_GCC_Construction_2019.pdf` | CPWD General Conditions of Contract |
| `CPWD_Specifications_Vol1_2019.pdf` | CPWD Specifications Volume 1 |
| `CPWD_Specifications_Vol2_2019.pdf` | CPWD Specifications Volume 2 |
| `CPWD_WorksManual_2019.pdf` | CPWD Works Manual |

These are issued by Bureau of Indian Standards (BIS) and the Central Public Works Department. They are public-authority publications used here for academic purposes only and are **not redistributed via this repository**.

Source the PDFs from:
- BIS catalogue: https://standards.bis.gov.in/
- CPWD: https://cpwd.gov.in/

### Build the index

```bash
python -m src.cli iscode build
```

This:

1. parses each PDF with PyMuPDF, splitting into section-level chunks
2. embeds each chunk with `gemini-embedding-001` (3072-dim)
3. stores everything in a ChromaDB collection at `.cache/chroma/iscode_corpus/`
4. builds a `{code_id::section: chunk_id}` registry used by Stage 4's citation verifier

Total: ~10,205 chunks, ~120 MB on disk. Build cost: ~₹15.

Verify:

```bash
python -m src.cli iscode summary
python -m src.cli iscode verify "IS 456:2000 Section 8.1"
python -m src.cli iscode query  "performance security bank guarantee"
```

## Run the synthetic demo (free)

```bash
# 1. Generate the synthetic tender (5 PDFs + 2 corrigenda)
python -m src.cli acquire synthesize --tender-id SYN_001

# 2. Full pipeline
python -m src.cli pipeline run --tender-id SYN_001

# 3. Render the report
python -m src.cli report html --tender-id SYN_001

# 4. Open the report
start corpus/SYN_001/reports/report.html  # Windows
# open corpus/SYN_001/reports/report.html  # macOS
```

End-to-end: ~5 minutes, ~₹3.

## Run on your own tender

```bash
# 1. Create the corpus directory
mkdir -p corpus/MY_TENDER

# 2. Drop your PDFs in
cp /path/to/tender_pdfs/*.pdf corpus/MY_TENDER/

# 3. Run the pipeline
python -m src.cli pipeline run --tender-id MY_TENDER

# 4. Render the report
python -m src.cli report html --tender-id MY_TENDER
```

For a 200-page tender, expect ~30–60 minutes of pipeline runtime and ~₹500–1500 of API cost. The cache makes re-runs free.

## Troubleshooting

### `429 RESOURCE_EXHAUSTED`

Your AI Studio free-tier rate limit is hit. Options:

1. Wait — limits reset minute-by-minute and day-by-day.
2. Drop to Flash for stages currently using Pro: `--score-model gemini-2.5-flash` etc.
3. Upgrade to paid tier in AI Studio.
4. Switch to Vertex AI (`GEMINI_USE_VERTEX=1`) — uses GCP billing instead of AI Studio quotas.

### `403 PERMISSION_DENIED ... API_KEY_SERVICE_BLOCKED`

Your API key has API restrictions that block `generativelanguage.googleapis.com`. Either remove the restriction in Cloud Console → APIs & Services → Credentials, or create a new key without restrictions.

### `Your project has exceeded its monthly spending cap`

AI Studio per-project monthly cap (default ₹200). Raise it at https://aistudio.google.com/u/0/spend — set a sensible cap based on your test budget.

### `gcp-vertex-sa.json was not found`

You set `GEMINI_USE_VERTEX=1` but haven't created a service account yet. Either:
- Set `GEMINI_USE_VERTEX=0` and use the AI Studio API key path
- Or follow the Vertex setup in the README

### Smoke test passes but real pipeline fails on `RESOURCE_EXHAUSTED`

The smoke test uses `gemini-2.5-flash`, which has a separate quota from `gemini-2.5-pro`. Run a Pro test:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from src.llm.gemini_client import get_default_client; c=get_default_client(); print(c.generate('Say yes', model='gemini-2.5-pro', force_refresh=True))"
```

If Pro returns `0 quota`, the project needs paid tier or Vertex.
