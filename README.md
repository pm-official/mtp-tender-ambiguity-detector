# Tender Ambiguity Detector

**Vagueness detection in Indian construction tender documents using the Bidder Commitment Test (BCT).**

> M.Tech thesis project, Construction Technology and Management, IIT Bombay (Department of Civil Engineering).
> Author: Prakhar (Roll No. 24M0631). Final defense: 25 June 2026.

[![tests](https://img.shields.io/badge/tests-246%20passing-green)](#testing)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](#setup)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## What it does

You drop a tender package (the bundle of NIT, GCC, SCC, Specifications, BoQ, and corrigenda PDFs) into the tool, pick which document and page range you want analysed, and the pipeline produces an audited list of vague clauses with rewritten alternatives:

1. **Detects** clauses where a bidder cannot work out a specific quantity, method, or applicable standard from the text alone (the **Bidder Commitment Test**).
2. **Adjudicates** apparent vs real vagueness by hybrid retrieval over the rest of the tender — many "vague" clauses are actually resolved later in another document.
3. **Critiques and severity-scores** the surviving real-vagueness flags. Each flag is tagged HIGH / MEDIUM / LOW based on commercial exposure × dispute likelihood × reviewer cost.
4. **Rewrites** confirmed-high flags grounded in a verified IS-code / CPWD section, with three independent guardrails:
   - **G1 citation integrity** — every cited IS code section is verified against the local IS-code corpus
   - **G2 differential BCT** — the rewrite must not leave more `CANNOT_DETERMINE` fields than the original
   - **G3 intent preservation** — embedding cosine similarity ≥ 0.6 vs the original

Each flag arrives with a complete audit trail (cached LLM responses, retrieved evidence, severity scores) and a draft pre-bid clarification letter ready to send to the Authority.

## Methodology in one diagram

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Stage 1    │──▶│   Stage 2    │──▶│   Stage 3    │──▶│   Stage 4    │
│              │   │              │   │              │   │              │
│   BCT        │   │  Hybrid      │   │  Critique +  │   │  IS-code-    │
│  (Flash)     │   │  retrieval + │   │  severity    │   │  grounded    │
│              │   │  REAL vs     │   │  (Pro)       │   │  rewrite     │
│   "What can  │   │  APPARENT    │   │              │   │  (Pro) with  │
│   a bidder   │   │  critic      │   │  HIGH/MED/   │   │  3 guard-    │
│   commit to?"│   │  (Flash)     │   │  LOW tier    │   │  rails       │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  candidate_         stage2_           stage3_           stage4_
  flags.jsonl        verdicts.jsonl    confirmed.jsonl   rewrites.jsonl
```

The headline contribution is **the Bidder Commitment Test**: instead of asking an LLM "is this clause ambiguous?" (a noisy classification problem), we ask "if you were a bidder, what specific commitment would you make?" — a structured elicitation problem with a deterministic decision rule. A flag is raised iff any of `{quantity, method, standard}` comes back as the sentinel string `CANNOT_DETERMINE`.

## Headline numbers (synthetic test, SYN_001)

Hand-labelled ground truth: 13 vague-by-design clauses, 13 precise-by-design clauses.

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Lexicon baseline (37 regex patterns) | 1.000 | 0.846 | 0.917 |
| BCT pipeline (Stage 1 only) | 0.611 | 0.846 | 0.710 |
| BCT + Stage 2 critic | 0.636 | 0.538 | 0.583 |
| **BCT + Stage 3 confirm** | **1.000** | 0.231 | 0.375 |

Cohen's κ between BCT and lexicon = 0.20 (moderate agreement — methods are complementary).

The full ablation, per-stage cost report, and explanation of every number is rendered in `corpus/SYN_001/reports/report.html` after running the pipeline.

## Repository layout

```
tender-ambiguity-detector/
├── pyproject.toml                # project + dev deps
├── .env.example                  # copy to .env, add your Gemini API key
├── .gitignore
├── LICENSE                       # MIT
├── README.md                     # this file
├── src/
│   ├── llm/gemini_client.py      # Gemini wrapper: caching, retry, cost tracking
│   ├── acquire/synthetic.py      # SYN_001 generator (5 PDFs + 2 corrigenda)
│   ├── parse/                    # PyMuPDF parser + clause-aware chunker
│   ├── extract/                  # definitions, quantities, references, vector index
│   ├── score/stage1_bct.py       # Bidder Commitment Test (Flash)
│   ├── retrieve/hybrid.py        # 3-route hybrid retrieval (R1+R2+R3) + RRF
│   ├── critic/stage2.py          # Stage 2: REAL vs APPARENT critic (Flash)
│   ├── critic/stage3.py          # Stage 3: critique + severity (Pro)
│   ├── rewrite/stage4.py         # Stage 4: IS-code-grounded rewrite + 3 guardrails (Pro)
│   ├── iscode/                   # IS-code corpus parser + index + citation verifier
│   ├── eval/                     # lexicon baseline, metrics, ablation, ground truth
│   ├── audit/run_manifest.py     # per-run JSON audit trail
│   ├── pipeline.py               # one-shot orchestrator
│   ├── pipeline_cli.py           # `pipeline run / status / history` CLI
│   ├── report/render.py          # HTML report renderer
│   ├── schemas/                  # Pydantic models for every stage
│   └── cli.py                    # top-level CLI entry point
├── resources/
│   └── iscode/                   # 37 IS-code + CPWD PDFs (gitignored, see docs/setup.md)
├── corpus/
│   ├── SYN_001/                  # synthetic tender + parsed artefacts (committed for demo)
│   └── JK_001/                   # real BMTPC tender (gitignored)
├── docs/
│   ├── methodology.md            # BCT formalism, why each stage, why each guardrail
│   ├── architecture.md           # how the pipeline pieces fit together
│   └── setup.md                  # detailed install + IS-code corpus build
├── scripts/                      # one-off analysis scripts
└── tests/                        # 246 offline tests + 1 live smoke test
```

## Setup

### One-time install

```bash
# 1. Python 3.11+ virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install in editable mode with dev extras
pip install -e ".[dev]"

# 3. Copy env template; add your Gemini key
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
# ...then edit .env and replace YOUR_GEMINI_API_KEY_HERE
```

Get a free Gemini API key at https://aistudio.google.com/app/apikey.

### Verify install

```bash
# Offline tests — no API call required
pytest tests/ -q

# Live smoke test — confirms your API key works
pytest tests/test_gemini_smoke.py -v
```

If 246 offline tests pass plus the smoke test, the build is good.

### Build the IS-code corpus index (one-time, ~2 minutes, ~₹15 of API spend)

The IS-code rewriter needs a local index of IS / CPWD section text. Drop the source PDFs into `resources/iscode/` (filenames listed in `docs/setup.md`) and run:

```bash
python -m src.cli iscode build
```

This produces a 10,205-chunk ChromaDB collection used by Stage 4.

## Running the pipeline

### Synthetic demo (no real money)

```bash
# Build the synthetic tender (one-time)
python -m src.cli acquire synthesize --tender-id SYN_001

# Run the full pipeline
python -m src.cli pipeline run --tender-id SYN_001

# Render the HTML report
python -m src.cli report html --tender-id SYN_001

# Open the report
start corpus/SYN_001/reports/report.html  # Windows
# open corpus/SYN_001/reports/report.html  # macOS
```

### Run on your own tender

```bash
# 1. Drop your PDFs into corpus/<your_tender_id>/
mkdir corpus/MY_TENDER
cp /path/to/*.pdf corpus/MY_TENDER/

# 2. Parse + extract + score + critique + confirm + rewrite
python -m src.cli pipeline run --tender-id MY_TENDER

# 3. Generate the report
python -m src.cli report html --tender-id MY_TENDER
```

### Evaluation

```bash
# Lexicon baseline + metrics
python -m src.cli eval lexicon  --tender-id SYN_001
python -m src.cli eval metrics  --tender-id SYN_001 --ground-truth synthetic

# Per-stage ablation (precision/recall by stage)
python -m src.cli eval ablation --tender-id SYN_001

# BCT-vs-lexicon side-by-side
python -m src.cli eval compare  --tender-id SYN_001
```

## Cost discipline

Every LLM call goes through `src/llm/gemini_client.py`, which:

- **caches** every response on disk by SHA-256 of the input — re-runs are free
- **tracks** input/output tokens and computes USD/INR cost per call
- **retries** transient errors (429, 503) with exponential backoff
- **routes** by model: Stage 1 BCT and Stage 2 critic on Flash (4× cheaper); Stage 3 critique + Stage 4 rewrite on Pro (where reasoning quality justifies the cost)

A full SYN_001 pipeline run including Stage 4 rewrites costs roughly **₹3** (~$0.04). A full real-tender run on the order of a few thousand clauses costs roughly ₹500–₹1500 (~$6–$18) depending on how many clauses survive Stage 1 to reach Pro-model stages.

## Citation discipline

Every IS-code citation produced by the rewrite stage is verified against the local IS-code registry (`src/iscode/citation.py`). If the cited (code, version, section) tuple does not exist in the registry, **G1 fails** and the rewrite is rejected — the system never silently produces an unverified citation.

The IS-code corpus currently includes:
IS 269, IS 383, IS 456, IS 800, IS 808, IS 875, IS 1200 (Parts 1–25), IS 1786, IS 1893, IS 13920, plus CPWD GCC 2019, CPWD Specifications 2019 (Volumes 1–2), and CPWD Works Manual 2019.

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — the Bidder Commitment Test formalised; rationale for each pipeline stage; rationale for each guardrail
- [`docs/architecture.md`](docs/architecture.md) — how Pydantic schemas, the LLM cache, the run-manifest layer, and the orchestrator fit together
- [`docs/setup.md`](docs/setup.md) — detailed installation; sourcing the IS-code corpus PDFs; troubleshooting

## Testing

```bash
# Full offline suite (no API calls)
pytest tests/ -q

# A specific stage
pytest tests/test_rewrite_stage4.py -v

# With the live API
pytest tests/test_gemini_smoke.py -v
```

Coverage spans schemas, parser, citation verifier, lexicon, metrics, ablation, all four pipeline stages, and the orchestrator.

## License

MIT — see [LICENSE](LICENSE). Tender PDFs in `corpus/JK_001/` and IS-code PDFs in `resources/iscode/` are public-authority documents but are not redistributed via this repository.

## Citation

If you use this work in academic publications, please cite:

```bibtex
@mastersthesis{prakhar2026bct,
  title  = {Bidder Commitment Test: A retrieval-grounded methodology for
            detecting and rewriting vague clauses in Indian construction
            tender documents},
  author = {Prakhar},
  school = {Indian Institute of Technology Bombay,
            Department of Civil Engineering},
  year   = {2026}
}
```

## Acknowledgements

This work was carried out at the Department of Civil Engineering, IIT Bombay, as part of the M.Tech in Construction Technology and Management. The IS-code corpus is reproduced from Bureau of Indian Standards publications for academic use only.
