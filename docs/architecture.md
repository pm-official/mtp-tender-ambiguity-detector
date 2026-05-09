# Architecture

## Module map

```
src/
├── llm/gemini_client.py        # Gemini wrapper: cache + retry + cost
├── schemas/                    # Pydantic models for every artefact
│   ├── clause.py               # Clause, BBox, PageInfo, DocumentTree
│   ├── structures.py           # Definition, Quantity, Reference
│   ├── bct.py                  # Commitment, BCTOutput, CandidateFlag
│   ├── stage2.py               # Stage2Verdict, HybridHit
│   ├── stage3.py               # ConfirmedFlag, SeverityScore
│   └── rewrite.py              # ISCodeChunk, ISCodeCitation, Rewrite
├── acquire/                    # Tender acquisition (synthetic + scrapers)
├── parse/                      # PDF → clauses.jsonl
├── extract/                    # definitions, quantities, references, vector index
├── score/                      # Stage 1 BCT
├── retrieve/                   # 3-route hybrid retrieval + RRF
├── critic/                     # Stage 2, Stage 3
├── rewrite/                    # Stage 4 + 3 guardrails
├── iscode/                     # IS-code corpus parser, index, citation verifier
├── eval/                       # lexicon, metrics, ablation, ground truth
├── audit/run_manifest.py       # per-run JSON audit trail
├── pipeline.py                 # one-shot orchestrator
├── pipeline_cli.py             # 'pipeline run / status / history' CLI
├── report/render.py            # HTML report renderer
└── cli.py                      # top-level CLI entry point
```

## Data flow

A tender lives in `corpus/<tender_id>/`:

```
corpus/<tender_id>/
├── *.pdf                              # raw input documents
├── manifest.json                      # acquisition manifest
├── parsed/
│   ├── clauses.jsonl                  # Stage 0 output
│   ├── document_tree.json
│   ├── definitions.jsonl
│   ├── quantities.jsonl
│   ├── references.jsonl
│   ├── reference_graph.json
│   ├── candidate_flags.jsonl          # Stage 1 output
│   ├── stage1_summary.json
│   ├── stage2_verdicts.jsonl          # Stage 2 output
│   ├── stage2_summary.json
│   ├── stage3_confirmed.jsonl         # Stage 3 output
│   ├── stage3_summary.json
│   ├── stage4_rewrites.jsonl          # Stage 4 output
│   ├── stage4_summary.json
│   ├── baseline_lexicon.jsonl         # eval: lexicon baseline
│   ├── metrics_synthetic.json         # eval: precision/recall vs GT
│   └── ablation.json                  # eval: per-stage ablation
├── runs/                              # per-run audit manifests
│   └── <timestamp>__<stage>__<hash>.json
└── reports/
    └── report.html                    # final user-facing artefact
```

Each stage reads its input JSONL, processes records, writes a new JSONL plus a summary JSON, and appends a manifest to `runs/`. No stage modifies prior outputs — the pipeline is fully append-only at the corpus level.

## The Gemini client

`src/llm/gemini_client.py` is the single point of contact with the Gemini API. Every other module calls `client.generate(...)` or `client.embed(...)`.

Responsibilities:

- **Backend selection** — Vertex AI (credit-eligible) or AI Studio API key (card-billed), via `GEMINI_USE_VERTEX` env var
- **Caching** — SHA-256-keyed disk cache, content-addressed by `(model, prompt, schema, temp, system_instruction, max_output_tokens)`. Identical re-calls are free.
- **Retry** — `tenacity` exponential backoff on transient errors (429, 503, RemoteProtocolError)
- **Cost tracking** — every call returns an `LLMCallStats` record with prompt/completion tokens, USD cost (computed from per-model rates), INR equivalent, latency, cache-hit flag
- **Structured output** — pass a `response_schema` (Pydantic model) and get back a typed instance, not a string

This abstraction lets every other module stay model-agnostic — they ask for a structured output of a given schema; the client handles the rest.

## Schemas

Every cross-module data structure is a Pydantic model. The decision rule is: if a value travels between modules or to/from disk, it has a schema.

Key schemas:

- `Clause` — one chunk of one document, with section path and bounding box
- `Commitment` (BCT output) — `obligation`, `quantity`, `method`, `standard`, `missing_info[]`
- `CandidateFlag` — Stage 1 output: clause + BCT output + flagged bit + reason
- `Stage2Verdict` — REAL / APPARENT + rationale + retrieved chunks
- `ConfirmedFlag` — Stage 3 output: critique verdict + severity tier + rationale
- `Rewrite` — Stage 4 output: rewrite text + IS-code citations + guardrail outcomes + status

Schemas live in `src/schemas/`. They are imported by every stage and are the public contract between stages.

## The run manifest

`src/audit/run_manifest.py` writes a JSON manifest per pipeline run to `corpus/<id>/runs/`:

```json
{
  "tender_id": "SYN_001",
  "stage": "stage_4_rewrite",
  "started_at": "2026-05-09T11:46:47Z",
  "completed_at": "2026-05-09T11:51:57Z",
  "models": {"rewrite": "gemini-2.5-pro", "embedding": "gemini-embedding-001"},
  "parameters": {"max_attempts": 3, "intent_threshold": 0.6, ...},
  "inputs":  [{"path": "candidate_flags.jsonl", "sha256": "..."}],
  "outputs": [{"path": "stage4_rewrites.jsonl", "sha256": "..."}],
  "llm_calls": [{"model": "gemini-2.5-pro", "tokens_in": 5234, "tokens_out": 312, "cost_usd": 0.014}, ...],
  "totals": {"calls": 27, "tokens_in": 41203, "tokens_out": 1924, "cost_usd": 0.087}
}
```

Combined with the on-disk LLM cache, any past run is exactly reproducible.

## CLI

`src/cli.py` wires every module's CLI through `argparse` subparsers:

```
python -m src.cli verify                                    # smoke test
python -m src.cli acquire   {init,drop,synthesize,scrape,list}
python -m src.cli parse     {run,inspect}
python -m src.cli extract   {run,query,summary}
python -m src.cli iscode    {build,query,verify,summary}
python -m src.cli score     {run,inspect}
python -m src.cli critique  {run,inspect}
python -m src.cli confirm   {run,inspect}
python -m src.cli rewrite   {run,inspect}
python -m src.cli pipeline  {run,status,history}
python -m src.cli report    {html}
python -m src.cli eval      {lexicon,metrics,ablation,compare}
```

Each subcommand is implemented in its module's `cli.py` file and registered via `add_subparser(sub)`.

## Tests

```
tests/
├── test_gemini_client_offline.py
├── test_gemini_smoke.py            # live API test (skipped without key)
├── test_clause_schema.py
├── test_pdf_parser.py
├── test_iscode_parser.py
├── test_iscode_citation.py
├── test_score_stage1_bct.py
├── test_critic_stage2.py
├── test_critic_stage3.py
├── test_rewrite_stage4.py
├── test_eval_lexicon.py
├── test_eval_metrics.py
├── test_eval_ablation.py
├── test_eval_synthetic_groundtruth.py
└── ...
```

246 offline tests + 1 live smoke test as of Session 11.

## HTML report

`src/report/render.py` produces a single-file `report.html` containing:

1. **Top-line summary** — clauses parsed, BCT flags, Stage 2/3 verdicts, severity tier counts
2. **Pre-bid clarification letter** — sorted by severity, ready to copy-paste and send to the Authority
3. **CONFIRMED ambiguities** — full clause text + Stage 2 rationale + Stage 3 critique + severity scores + Stage 4 rewrite + guardrail status
4. **REJECTED-on-Stage-3** — clauses where Stage 3 overruled Stage 2
5. **APPARENT ambiguities** — clauses resolved by retrieval
6. **Evaluation section** — precision/recall + ablation + agreement, with explanations
7. **Footer** — methodology + author

The report is self-contained (CSS inlined, no external assets) and renders offline.
