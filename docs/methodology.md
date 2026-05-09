# Methodology

## The problem

Indian construction tender documents — Notice Inviting Tender, General Conditions of Contract, Special Conditions of Contract, Specifications, Bill of Quantities — routinely contain clauses whose contractual scope cannot be determined by reading the clause alone. Examples from real BMTPC and CPWD tenders:

- *"Performance security in a form acceptable to the Authority"* — what form?
- *"Materials of suitable quality from approved sources"* — quality criteria? approval process?
- *"Quality control testing as per relevant IS code with sufficient frequency"* — which IS code? what frequency?
- *"Local labour wherever feasible"* — what fraction? feasibility test?

A bidder reading these has to either (a) price in a contingency for the worst-case interpretation, (b) raise a pre-bid query (slow, often non-binding), or (c) gamble. All three reduce competition and inflate prices. Disputes during execution traceable to such ambiguity drive a meaningful share of construction-claim litigation.

This project produces a tool that detects these clauses, scores their severity, and proposes IS-code-grounded rewrites with verified citations.

## The Bidder Commitment Test (BCT)

The headline contribution. Instead of asking an LLM "is clause X ambiguous?" — a noisy classification problem — BCT asks the LLM to **role-play a bidder** and produce a structured commitment record:

```python
class Commitment(BaseModel):
    obligation: str               # what the bidder is being asked to do
    quantity:   str | "CANNOT_DETERMINE"
    method:     str | "CANNOT_DETERMINE"
    standard:   str | "CANNOT_DETERMINE"
    missing_info: list[str]       # questions a bidder would ask
```

The decision rule is deterministic from the structured output: **the clause is flagged if any of `{quantity, method, standard}` equals the sentinel string `CANNOT_DETERMINE`**.

This framing has three advantages over direct ambiguity classification:

1. **Auditable** — every flag carries the structured Commitment record. A reviewer can see *why* the flag fired (which field is missing) without reading prompt traces.
2. **Generative-not-discriminative** — the LLM is producing content (a commitment), not classifying. Modern LLMs are better at the former.
3. **Naturally produces pre-bid questions** — the `missing_info` field is exactly what a bidder would ask the Authority. We use these to draft a clarification letter.

## The four-stage pipeline

### Stage 1 — BCT scoring

Per-clause Bidder Commitment Test on Gemini Flash. Pre-filter skips tables and tiny chunks. Obligation classifier skips clauses that aren't imperative obligations (definitions, scope statements, etc.). Survivors get the full BCT prompt.

Output: `candidate_flags.jsonl` with one record per clause.

Why Flash? BCT is a structured-extraction task with a clear schema. Pro's extra reasoning isn't needed and costs 4× more.

### Stage 2 — Hybrid retrieval + REAL-vs-APPARENT critic

Many "vague" clauses are actually resolved later in the same package — e.g. SCC says "the Engineer" without defining the role, but GCP §4.1 defines "the Engineer". This is **apparent vagueness**.

For each Stage-1 flag, three retrieval routes search the rest of the tender:

- **R1 dense semantic** — vector similarity over all clauses (gemini-embedding-001, 3072-dim)
- **R2 shared defined-term** — clauses that re-use a defined term from this clause
- **R3 shared canonical referent** — clauses that point to the same external authority/document

Results are fused with **Reciprocal Rank Fusion** (k=60). The top-K hits + the original clause go to a Flash critic that returns one of:

- **REAL** — vagueness persists after retrieval; bidder still cannot commit
- **APPARENT** — retrieval resolves the gap; clause is fine in context

Output: `stage2_verdicts.jsonl`.

Why Flash here? Binary classification on retrieved evidence — not deep reasoning.

### Stage 3 — Critique + severity scoring

REAL flags from Stage 2 go through a Pro-quality critique that:

1. **Confirms or rejects** the REAL verdict (CONFIRMED / REJECTED). Stage 2 is a Flash-quality first pass; Stage 3 is the high-confidence second pass.
2. **Severity-scores** confirmed flags on three axes:
   - `commercial_exposure` (1–5) — how much rupee value rides on the resolution
   - `dispute_likelihood` (1–5) — how likely is a downstream dispute
   - `reviewer_cost` (1–5) — manual review effort
   - composite = sum of the three; tier = HIGH (≥10) / MEDIUM (≥6) / LOW

Output: `stage3_confirmed.jsonl`.

Why Pro? Severity scoring is a judgment call requiring nuanced reasoning across legal, contractual, and engineering considerations. Quality justifies the cost.

### Stage 4 — IS-code-grounded rewrite + three guardrails

For each CONFIRMED HIGH flag, Stage 4 generates a rewrite that:

- cites at least one IS / CPWD section
- fills the specific gaps named by Stage 1 BCT
- preserves contractual intent (no scope expansion, no liability shift)

Three independent guardrails check the rewrite:

- **G1 Citation integrity** — parse every IS/CPWD citation, verify (code, version, section) against the local IS-code registry. Any unresolved citation → reject.
- **G2 Differential BCT** — feed the rewrite back through Stage 1 BCT. The rewrite must not have *more* `CANNOT_DETERMINE` fields than the original. This is a **non-regression** check, not a perfect-spec check, because Stage 1 BCT's obligation decomposition can vary between original and rewrite (a rewrite may extract a continuation property like "valid until 90 days after DLP" as its own sub-obligation).
- **G3 Intent preservation** — embedding cosine similarity between rewrite and original ≥ 0.6. Below that threshold, the rewrite has changed contractual scope and is rejected.

Up to three retry attempts with progressively stricter prompts naming the previous failure mode.

Output: `stage4_rewrites.jsonl`.

Why Pro? Multi-constraint generation (citation + IS-code semantics + intent preservation) where reasoning matters.

## Evaluation

### Synthetic tender — SYN_001

We generate a synthetic tender (`src/acquire/synthetic.py`) containing a deliberate mix of vague and precise clauses. Hand-labelled in `src/eval/synthetic_groundtruth.py` — 13 vague-by-design + 13 precise-by-design = 26 labelled clauses across NIT, GCC, SCC, and Specifications.

### Lexicon baseline

A 37-pattern regex baseline (`src/eval/lexicon_baseline.py`) detects classical vagueness markers across six categories: *subjectivity* ("reasonable", "adequate"), *effort_pledges* ("best efforts", "due diligence"), *delegated_authority* ("acceptable to the Authority", "as directed"), *approximation*, *feasibility_dodge* ("wherever feasible"), and *underspecified_quantifier* ("regular intervals", "sufficient frequency").

### Ablation rungs

`src/eval/ablation.py` reports precision/recall at five filtering levels:

- **A0 No-filter** — flag every clause (gives the universe rate)
- **A1 Lexicon** — regex baseline
- **A2 BCT only** — Stage 1 alone
- **A3 + Stage 2** — keep only flags Stage 2 marked REAL
- **A4 + Stage 3** — keep only flags Stage 3 marked CONFIRMED

The expected pattern is precision climbing A2 → A4 while recall holds or drops slightly — quantifies "more pipeline, fewer false positives".

### Cohen's κ

Reported between BCT (Stage 1) and lexicon baseline as a reference-free agreement metric. Values around 0.2 indicate the methods are catching genuinely different clauses — neither subsumes the other.

## Why no manual annotation?

The annotation budget for vagueness in tender clauses is prohibitive — every label needs a domain expert reading the surrounding clauses. We make do with two cheaper substitutes:

1. **Synthetic ground truth** for SYN_001 — we wrote the tender, so we know which clauses are vague-by-design.
2. **Corrigendum-derived silver labels** for real tenders — when an Authority issues a corrigendum amending a clause, that's a ground-truth signal that the original clause was unclear (`src/label/silver.py`, planned).

This is documented as a paper limitation. Real-world precision claims are bounded by the silver-label noise rate, not zero.

## Reproducibility

Every pipeline run writes a JSON manifest to `corpus/<id>/runs/<timestamp>__<stage>__<hash>.json` containing:

- model name(s), version(s), and decoding parameters
- environment vars relevant to the run
- input file hashes
- per-call telemetry (tokens, cost, latency)
- output file path and hash

This is enforced by `src/audit/run_manifest.py`. Combined with the on-disk LLM response cache, any past run can be reproduced exactly given the same inputs.
