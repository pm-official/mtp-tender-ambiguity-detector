"""Streamlit demo app for the Tender Ambiguity Detector.

Two modes:
  - Frozen demo: load pre-computed SYN_001 outputs, walk through the four
    stages with inline explanations. Zero API cost. Best for first-time
    visitors and defense walkthrough.
  - Live: upload a tender package, pick a document + page range, run the
    full BCT pipeline live with real-time stage-by-stage progress.

Every metric and decision in the UI has an "ℹ️ Explain" toggle. The goal
is that a viva examiner can answer their own questions by reading what's
on screen — no black boxes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

# Suppress noisy loggers
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tender Ambiguity Detector — BCT pipeline",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
CORPUS_ROOT = PROJECT_ROOT / "corpus"
DEMO_TENDER_ID = "SYN_001"
MAX_RECENT_RUNS = 3

# Read API key from Streamlit secrets (cloud) or .env (local)
if "GEMINI_API_KEY" not in os.environ:
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        # Fall through to dotenv loaded by gemini_client
        pass


# ─── Explainer helper ──────────────────────────────────────────────────────
def explain(label: str, body: str, *, expanded: bool = False) -> None:
    """Render an inline ℹ️ Explain expander next to a metric/decision.

    Used relentlessly throughout the app to make sure no number is
    presented without an explanation of what it means and how it was
    computed.
    """
    with st.expander(f"ℹ️ {label}", expanded=expanded):
        st.markdown(body)


# ─── Cached resources ──────────────────────────────────────────────────────
@st.cache_resource
def get_gemini_client():
    """Process-wide GeminiClient. Cached so we don't re-init on every rerun."""
    from src.llm.gemini_client import get_default_client
    return get_default_client()


@st.cache_data
def load_demo_artefacts():
    """Load all SYN_001 artefacts into memory once."""
    parsed = CORPUS_ROOT / DEMO_TENDER_ID / "parsed"
    return {
        "clauses":      _read_jsonl(parsed / "clauses.jsonl"),
        "candidates":   _read_jsonl(parsed / "candidate_flags.jsonl"),
        "stage2":       _read_jsonl(parsed / "stage2_verdicts.jsonl"),
        "stage3":       _read_jsonl(parsed / "stage3_confirmed.jsonl"),
        "stage4":       _read_jsonl(parsed / "stage4_rewrites.jsonl"),
        "lexicon":      _read_jsonl(parsed / "baseline_lexicon.jsonl"),
        "ablation":     _read_json(parsed / "ablation.json"),
        "metrics":      _read_json(parsed / "metrics_synthetic.json"),
        "stage1_summ":  _read_json(parsed / "stage1_summary.json"),
        "stage2_summ":  _read_json(parsed / "stage2_summary.json"),
        "stage3_summ":  _read_json(parsed / "stage3_summary.json"),
        "stage4_summ":  _read_json(parsed / "stage4_summary.json"),
    }


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ─── Session-state init ─────────────────────────────────────────────────────
if "recent_runs" not in st.session_state:
    st.session_state.recent_runs = []   # list of dicts: {tender_id, doc_id, pages, run_id, summary}
if "current_run" not in st.session_state:
    st.session_state.current_run = None


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📑 BCT Pipeline")
    st.caption("Tender ambiguity detection — IIT Bombay M.Tech thesis")
    st.divider()

    mode = st.radio(
        "Mode",
        ["📂 Frozen demo (SYN_001)", "📤 Upload your tender (live)"],
        index=0,
        help="Frozen demo loads pre-computed results instantly. "
             "Live mode runs the pipeline on your uploaded PDFs.",
    )
    is_live_mode = mode.startswith("📤")

    st.divider()

    # ─── Live-mode controls ──────────────────────────────────────────────
    selected_doc = None
    page_start = 1
    page_end = 5
    uploaded_files = None
    run_clicked = False

    if is_live_mode:
        st.markdown("**Step 1 — upload PDFs**")
        uploaded_files = st.file_uploader(
            "Tender package",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload all PDFs that belong to one tender package "
                 "(NIT, GCC, SCC, Specifications, BoQ, corrigenda).",
        )

        if uploaded_files:
            st.markdown("**Step 2 — pick the document to analyse**")
            doc_names = [f.name for f in uploaded_files]
            selected_doc = st.selectbox(
                "Document",
                doc_names,
                help="Only this document will be scored. Other uploaded PDFs "
                     "are still indexed so the Stage-2 retriever can search "
                     "across the full package.",
            )

            st.markdown("**Step 3 — pick the page range**")
            # Quick estimate of total pages — use PyMuPDF
            try:
                import fitz
                file_obj = next(f for f in uploaded_files if f.name == selected_doc)
                file_obj.seek(0)
                doc = fitz.open(stream=file_obj.read(), filetype="pdf")
                total_pages = len(doc)
                doc.close()
                file_obj.seek(0)
            except Exception:
                total_pages = 100

            col_a, col_b = st.columns(2)
            with col_a:
                page_start = st.number_input(
                    "From page", min_value=1, max_value=total_pages, value=1, step=1,
                )
            with col_b:
                page_end = st.number_input(
                    "To page", min_value=page_start, max_value=total_pages,
                    value=min(5, total_pages), step=1,
                )
            st.caption(f"Document has **{total_pages}** pages. "
                       f"Selecting **{page_end - page_start + 1}** pages "
                       f"(p.{page_start}–p.{page_end}).")

            st.divider()
            st.markdown("**Step 4 — run**")
            run_clicked = st.button(
                "🚀 Run full pipeline (Stages 1 → 4)",
                use_container_width=True,
                type="primary",
            )
            st.caption(
                "This will call the Gemini API. The configured key bills against "
                "the project owner's account."
            )

    st.divider()

    # ─── Recent runs ─────────────────────────────────────────────────────
    st.markdown("**Recent runs**")
    if not st.session_state.recent_runs:
        st.caption("_(no runs in this session yet)_")
    else:
        for i, r in enumerate(st.session_state.recent_runs):
            label = (
                f"📄 {r['doc_id']} pp.{r['page_start']}-{r['page_end']} "
                f"· {r['summary']['total_flags']} flags · "
                f"₹{r['summary']['total_cost_inr']:.2f}"
            )
            if st.button(label, key=f"recent_{i}", use_container_width=True):
                st.session_state.current_run = r
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.title("Tender Ambiguity Detector")
st.markdown(
    "**Vagueness detection in Indian construction tenders using the "
    "Bidder Commitment Test (BCT).** "
    "M.Tech thesis project, Department of Civil Engineering, IIT Bombay."
)

explain(
    "What this tool does (and what BCT is)",
    """
**The problem.** Indian construction tender documents (NIT, GCC, SCC, Specs, BoQ)
routinely contain clauses where a bidder cannot work out a specific quantity,
method, or applicable standard from the text alone. These clauses cause
disputes during execution.

**The Bidder Commitment Test (BCT)** is the headline contribution of this
thesis. Instead of asking an LLM "is this clause ambiguous?" — a noisy
classification problem — BCT asks the LLM to **role-play a bidder** and
produce a structured commitment:

```
{
  "obligation":   "...",
  "quantity":     "...  | CANNOT_DETERMINE",
  "method":       "...  | CANNOT_DETERMINE",
  "standard":     "...  | CANNOT_DETERMINE",
  "missing_info": ["What ...?", "How ...?"]
}
```

The decision rule is **deterministic**: a clause is flagged iff any of
`{quantity, method, standard}` returns the sentinel string
`CANNOT_DETERMINE`.

**Four stages** filter and improve those flags:

| Stage | Purpose | Model |
|---|---|---|
| 1 BCT scoring | Per-clause commitment elicitation | Flash (cheap) |
| 2 Hybrid retrieval critic | Is this **REAL** vs **APPARENT** vagueness? (Apparent = resolved later in the package.) | Flash |
| 3 Critique + severity | High-quality second-pass; commercial / dispute / reviewer-cost severity | Pro |
| 4 IS-code rewrite | Generate a rewrite citing real IS / CPWD sections | Pro + 3 guardrails |

This app shows what each stage produced and explains why every number is what it is.
""",
    expanded=False,
)

st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# LIVE PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
def run_live_pipeline(
    uploaded_files,
    selected_doc: str,
    page_start: int,
    page_end: int,
) -> dict:
    """Run the full pipeline live on uploaded PDFs and return results.

    Uses st.status() to show step-by-step progress.
    """
    from src.audit.run_manifest import start_run, finish_run, write_manifest

    # ─── Set up a unique tender directory ─────────────────────────────────
    run_id = f"UPLOAD_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    tender_dir = CORPUS_ROOT / run_id
    tender_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = tender_dir / "parsed"
    parsed_dir.mkdir(exist_ok=True)
    (tender_dir / "runs").mkdir(exist_ok=True)
    (tender_dir / "reports").mkdir(exist_ok=True)

    # Copy uploaded files into the tender dir
    for f in uploaded_files:
        f.seek(0)
        (tender_dir / f.name).write_bytes(f.read())

    summary = {
        "tender_id": run_id,
        "doc_id": selected_doc,
        "page_start": page_start,
        "page_end": page_end,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    client = get_gemini_client()

    with st.status("Pipeline running...", expanded=True, state="running") as status:

        # ─── Stage 0a: Parse ───────────────────────────────────────────
        st.write("**Stage 0a — parsing PDFs**")
        from src.parse.pipeline import parse_tender
        parse_summary = parse_tender(tender_dir)
        clauses = _read_jsonl(parsed_dir / "clauses.jsonl")
        clauses_in_scope = [
            c for c in clauses
            if c.get("doc_id") == selected_doc
            and page_start <= c.get("page", 1) <= page_end
        ]
        st.write(f"Parsed {len(clauses)} clauses across {len(uploaded_files)} document(s). "
                 f"**{len(clauses_in_scope)}** clauses fall in {selected_doc} "
                 f"pages {page_start}–{page_end} — these are scored next.")

        if not clauses_in_scope:
            status.update(label="No clauses in selected page range", state="error")
            st.error("No clauses found in the selected page range. "
                     "Try a wider range or a different document.")
            return None

        # ─── Filter clauses.jsonl to selected scope (Stages 1+ read it) ─
        # Save full clauses for retrieval; save filtered for scoring.
        full_clauses_path = parsed_dir / "clauses.jsonl"
        # Keep a copy of full clauses (Stage 2 retrieval will use this)
        full_clauses_backup = parsed_dir / "clauses_full.jsonl"
        shutil.copy(full_clauses_path, full_clauses_backup)
        # Write filtered clauses for Stage 1
        with full_clauses_path.open("w", encoding="utf-8") as fout:
            for c in clauses_in_scope:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")

        # ─── Stage 0b: Build vector index over FULL clause set ─────────
        st.write("**Stage 0b — building vector index** (for Stage 2 retrieval)")
        # Restore full clauses for indexing, then re-filter after.
        shutil.copy(full_clauses_backup, full_clauses_path)
        from src.extract.pipeline import extract_all
        try:
            extract_all(
                tender_dir,
                build_index=True,
                build_definitions=False,  # Skip — adds API cost without clear demo value
                build_quantities=False,
                build_references=False,
                classify_referents_with_llm=False,
                llm_definitions_fallback=False,
            )
            st.write("Vector index built.")
        except Exception as e:
            st.warning(f"Vector index step skipped ({e}). Stage 2 will still run "
                       "but with weaker retrieval evidence.")
        # Re-filter clauses for scoring
        with full_clauses_path.open("w", encoding="utf-8") as fout:
            for c in clauses_in_scope:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")

        # ─── Stage 1: BCT ────────────────────────────────────────────────
        st.write(f"**Stage 1 — Bidder Commitment Test** (Flash on {len(clauses_in_scope)} clauses)")
        from src.score.pipeline import run_stage1
        s1_summ = run_stage1(tender_dir, score_max=None, score_skip_filter=False)
        flagged_total = s1_summ.get("flagged", 0)
        st.write(f"Stage 1 done: **{s1_summ.get('eligible', 0)}** clauses scored, "
                 f"**{flagged_total}** with `CANNOT_DETERMINE`. "
                 f"Cost: ₹{s1_summ.get('total_cost_inr', 0):.2f}.")

        # ─── Restore full clauses for Stage 2 retrieval ─────────────────
        shutil.copy(full_clauses_backup, full_clauses_path)

        # ─── Stage 2: Hybrid critic ─────────────────────────────────────
        st.write("**Stage 2 — REAL vs APPARENT critic** (Flash + retrieval)")
        from src.critic.pipeline import run_stage2
        s2_summ = run_stage2(tender_dir, critique_max=None, critique_top_k=8)
        real_count = s2_summ.get("verdict_real", 0)
        st.write(f"Stage 2 done: "
                 f"**{real_count} REAL**, "
                 f"**{s2_summ.get('verdict_apparent', 0)} APPARENT**, "
                 f"**{s2_summ.get('verdict_weak', 0)} WEAK**. "
                 f"Cost: ₹{s2_summ.get('total_cost_inr', 0):.2f}.")

        # ─── Stage 3: Critique + severity ───────────────────────────────
        st.write("**Stage 3 — critique + severity** (Pro)")
        from src.critic.stage3_pipeline import run_stage3
        s3_summ = run_stage3(tender_dir, confirm_max=None, confirm_skip_severity=False)
        confirmed_count = s3_summ.get("critique_confirmed", 0)
        sev_dist = s3_summ.get("severity_distribution") or {}
        st.write(f"Stage 3 done: "
                 f"**{confirmed_count} CONFIRMED** "
                 f"({sev_dist.get('high', 0)} high, "
                 f"{sev_dist.get('medium', 0)} medium, "
                 f"{sev_dist.get('low', 0)} low), "
                 f"**{s3_summ.get('critique_rejected', 0)} REJECTED**. "
                 f"Cost: ₹{s3_summ.get('total_cost_inr', 0):.2f}.")

        # ─── Stage 4: Rewrite ───────────────────────────────────────────
        st.write("**Stage 4 — IS-code-grounded rewrite** (Pro + 3 guardrails)")
        from src.rewrite.pipeline import run_stage4
        try:
            s4_summ = run_stage4(
                tender_dir,
                rewrite_max=None,
                rewrite_only_severity="high",
            )
            st.write(f"Stage 4 done: "
                     f"**{s4_summ.get('status_counts', {}).get('ACCEPTED', 0)} ACCEPTED**, "
                     f"**{s4_summ.get('status_counts', {}).get('RETRY_LIMIT_REACHED', 0)} retry-limited**, "
                     f"**{s4_summ.get('status_counts', {}).get('ERROR', 0)} errored**. "
                     f"Cost: ₹{s4_summ.get('total_cost_inr', 0):.2f}.")
        except Exception as e:
            st.warning(f"Stage 4 skipped: {e}")
            s4_summ = {"status_counts": {}, "total_cost_inr": 0.0}

        # ─── Aggregate ──────────────────────────────────────────────────
        total_cost = (
            s1_summ.get("total_cost_inr", 0)
            + s2_summ.get("total_cost_inr", 0)
            + s3_summ.get("total_cost_inr", 0)
            + s4_summ.get("total_cost_inr", 0)
        )
        summary.update({
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "clauses_in_scope":   len(clauses_in_scope),
            "total_flags":        flagged_total,
            "real_flags":         real_count,
            "confirmed":          confirmed_count,
            "rewrites_accepted":  s4_summ.get("status_counts", {}).get("ACCEPTED", 0),
            "total_cost_inr":     total_cost,
            "total_cost_usd":     total_cost / 84.0,
        })

        status.update(label=f"✅ Pipeline complete · ₹{total_cost:.2f}", state="complete")

    return {
        "tender_id":   run_id,
        "tender_dir":  str(tender_dir),
        "doc_id":      selected_doc,
        "page_start":  page_start,
        "page_end":    page_end,
        "summary":     summary,
        "artefacts":   _load_run_artefacts(parsed_dir),
    }


def _load_run_artefacts(parsed_dir: Path) -> dict:
    return {
        "clauses":     _read_jsonl(parsed_dir / "clauses.jsonl"),
        "candidates":  _read_jsonl(parsed_dir / "candidate_flags.jsonl"),
        "stage2":      _read_jsonl(parsed_dir / "stage2_verdicts.jsonl"),
        "stage3":      _read_jsonl(parsed_dir / "stage3_confirmed.jsonl"),
        "stage4":      _read_jsonl(parsed_dir / "stage4_rewrites.jsonl"),
    }


# ─── Trigger live run if requested ────────────────────────────────────────
if is_live_mode and run_clicked:
    if not uploaded_files:
        st.error("Upload at least one PDF first.")
    else:
        result = run_live_pipeline(
            uploaded_files, selected_doc, int(page_start), int(page_end),
        )
        if result is not None:
            st.session_state.current_run = result
            # push onto recent runs (cap at MAX_RECENT_RUNS)
            st.session_state.recent_runs.insert(0, result)
            st.session_state.recent_runs = st.session_state.recent_runs[:MAX_RECENT_RUNS]


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════
def render_results(
    *,
    title: str,
    candidates: list,
    stage2: list,
    stage3: list,
    stage4: list,
    ablation: Optional[dict] = None,
    metrics: Optional[dict] = None,
    summary: Optional[dict] = None,
) -> None:
    """Render the four-stage results block. Used for both frozen + live."""
    st.header(title)

    # ─── Top-line summary cards ──────────────────────────────────────────
    cols = st.columns(5)
    cols[0].metric("Candidate flags (Stage 1)", sum(1 for c in candidates if c.get("flagged")))
    cols[1].metric("REAL (Stage 2)", sum(1 for v in stage2 if v.get("verdict") == "REAL"))
    cols[2].metric("CONFIRMED (Stage 3)", sum(1 for c in stage3 if c.get("critique_verdict") == "CONFIRMED"))
    cols[3].metric("ACCEPTED rewrites (Stage 4)", sum(1 for r in stage4 if r.get("status") == "ACCEPTED"))
    if summary and "total_cost_inr" in summary:
        cols[4].metric("Total cost", f"₹{summary['total_cost_inr']:.2f}")

    explain(
        "How to read these counts",
        """
The four counts narrow down the candidate set as the pipeline becomes more
confident — *recall-at-Stage-1, precision-at-Stage-3*:

- **Stage 1 candidates** is the recall-oriented set: every clause where the
  Bidder Commitment Test could not pin down quantity, method, **or** standard.
  Expected to over-flag; that's by design.
- **REAL after Stage 2** are clauses where the *rest of the tender package*
  doesn't resolve the gap. Many flags become **APPARENT** here — they're
  actually fine in context.
- **CONFIRMED after Stage 3** is the high-confidence set, double-checked by a
  Pro-quality model and severity-scored.
- **ACCEPTED rewrites** in Stage 4 are the rewrites that pass all three
  guardrails (citation integrity, BCT non-regression, intent preservation).
""",
    )
    st.divider()

    # ─── Stage 1: candidate flags ─────────────────────────────────────────
    st.subheader("Stage 1 — Bidder Commitment Test")
    explain(
        "What Stage 1 does",
        """
For each clause, the LLM (Flash) extracts a structured **Commitment**:
`{obligation, quantity, method, standard, missing_info[]}`.

A clause is flagged iff any of `{quantity, method, standard}` returns the
sentinel string `CANNOT_DETERMINE`. The decision rule is deterministic from
the structured output — no fuzzy classification.

The `missing_info` field doubles as the source for the *pre-bid clarification
letter* shown below.
""",
        expanded=False,
    )
    flagged = [c for c in candidates if c.get("flagged")]
    if not flagged:
        st.info("No clauses were flagged at Stage 1.")
    else:
        for c in flagged[:50]:  # cap display
            with st.expander(f"🟠 `{c.get('clause_id')}` — {c.get('flag_reason', 'flagged')}"):
                st.markdown(f"**Clause text:**")
                st.text(c.get("clause_text", "")[:1500])
                bct = c.get("bct_output", {}).get("commitments", [])
                if bct:
                    st.markdown("**BCT extracted commitments:**")
                    for cmt in bct:
                        cd_fields = [f for f in ("quantity", "method", "standard")
                                      if cmt.get(f) == "CANNOT_DETERMINE"]
                        st.markdown(
                            f"- *{cmt.get('obligation', '?')}* — "
                            f"missing: **{', '.join(cd_fields) or 'none'}**"
                        )
                        if cmt.get("missing_info"):
                            for q in cmt["missing_info"][:5]:
                                st.markdown(f"  - bidder would ask: *“{q}”*")
    st.divider()

    # ─── Stage 2: REAL vs APPARENT ────────────────────────────────────────
    st.subheader("Stage 2 — REAL vs APPARENT critic")
    explain(
        "What Stage 2 does",
        """
For each Stage-1 flag, the system retrieves the most relevant chunks from
the **rest of the tender package** using **hybrid retrieval** (three routes
fused with Reciprocal Rank Fusion):

- **R1 dense semantic** — vector similarity (gemini-embedding-001, 3072-dim)
- **R2 shared defined-term** — clauses re-using the same defined term
- **R3 shared canonical referent** — clauses pointing to the same external authority

A Flash critic then judges:

- **REAL** — vagueness persists; bidder still cannot commit even with the rest of the package
- **APPARENT** — retrieval resolves the gap; the clause is fine in context
- **WEAK** — borderline (default to REAL if escalating)

This is the layer that prevents the system from flagging clauses like *“the Engineer”*
that are genuinely defined elsewhere in the GCC.
""",
    )
    if not stage2:
        st.info("Stage 2 produced no verdicts.")
    else:
        verdict_counts = {}
        for v in stage2:
            verdict_counts[v.get("verdict", "?")] = verdict_counts.get(v.get("verdict", "?"), 0) + 1
        st.write(f"Verdict distribution: {verdict_counts}")
        real = [v for v in stage2 if v.get("verdict") == "REAL"]
        apparent = [v for v in stage2 if v.get("verdict") == "APPARENT"]

        if real:
            st.markdown("**REAL flags (vagueness persists):**")
            for v in real[:25]:
                with st.expander(f"🔴 `{v.get('clause_id')}`"):
                    st.markdown(f"**Rationale:** {v.get('rationale', '')}")
                    if v.get("retrieved_chunk_ids"):
                        st.markdown(f"**Retrieved clauses checked:** "
                                    f"{len(v['retrieved_chunk_ids'])} hits")
        if apparent:
            with st.expander(f"🟢 **APPARENT flags ({len(apparent)} resolved by retrieval)**"):
                for v in apparent[:25]:
                    st.markdown(f"- `{v.get('clause_id')}`: {v.get('rationale', '')[:200]}")
    st.divider()

    # ─── Stage 3: critique + severity ─────────────────────────────────────
    st.subheader("Stage 3 — critique + severity")
    explain(
        "What Stage 3 does",
        """
Stage 2 is a Flash-quality first pass. Stage 3 is a Pro-quality second pass that:

1. **Confirms or rejects** the REAL verdict — a high-quality reasoner
   sometimes overrules Stage 2 (REJECTED).
2. **Severity-scores** confirmed flags on three axes (1–5 each):
   - `commercial_exposure` — how much rupee value rides on the resolution
   - `dispute_likelihood` — how likely is a downstream dispute
   - `reviewer_cost` — manual review effort

The composite score = sum of the three. **Tier**:
- **HIGH** ≥ 10
- **MEDIUM** 6–9
- **LOW** ≤ 5

Stage 4 only rewrites **HIGH** flags (cost discipline).
""",
    )
    if not stage3:
        st.info("Stage 3 produced no records.")
    else:
        confirmed = [c for c in stage3 if c.get("critique_verdict") == "CONFIRMED"]
        rejected = [c for c in stage3 if c.get("critique_verdict") == "REJECTED"]

        if confirmed:
            st.markdown(f"**CONFIRMED ({len(confirmed)}):**")
            for c in sorted(confirmed, key=lambda c: -c.get("composite_severity", 0))[:20]:
                tier = c.get("severity_tier", "?")
                emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(tier, "⚪")
                with st.expander(
                    f"{emoji} `{c.get('clause_id')}` — "
                    f"**{tier.upper()}** severity ({c.get('composite_severity', 0)}/15)"
                ):
                    st.text(c.get("clause_text", "")[:800])
                    sev = c.get("severity") or {}
                    if sev:
                        st.markdown(
                            f"- commercial exposure: **{sev.get('commercial_exposure', '?')}/5**\n"
                            f"- dispute likelihood: **{sev.get('dispute_likelihood', '?')}/5**\n"
                            f"- reviewer cost: **{sev.get('reviewer_cost', '?')}/5**"
                        )
                        if sev.get("rationale"):
                            st.markdown(f"**Why:** {sev['rationale']}")
        if rejected:
            with st.expander(f"⚪ **REJECTED on Stage-3 critique ({len(rejected)})**"):
                for c in rejected[:15]:
                    st.markdown(f"- `{c.get('clause_id')}`: "
                                f"{c.get('critique_refutation', '')[:250]}")
    st.divider()

    # ─── Stage 4: rewrites ────────────────────────────────────────────────
    st.subheader("Stage 4 — IS-code-grounded rewrite")
    explain(
        "What Stage 4 does + the three guardrails",
        """
For each CONFIRMED HIGH flag, a Pro model proposes a rewrite that:

- cites at least one IS / CPWD section
- fills the specific gaps named by Stage 1's BCT
- preserves contractual intent (no scope expansion)

**Three independent guardrails** check the rewrite:

- **G1 Citation integrity** — every cited (code, version, section) is verified
  against the local IS-code registry. Hallucinated citations are caught here.
- **G2 Differential BCT** — the rewrite is fed back through Stage 1 BCT. The
  rewrite must not have *more* `CANNOT_DETERMINE` fields than the original.
  (Non-regression check — Stage 1's obligation decomposition can vary, so a
  zero-CD requirement would be too strict.)
- **G3 Intent preservation** — embedding cosine similarity ≥ 0.6. Below that,
  the rewrite has changed contractual scope and is rejected.

Up to 3 retries with progressively stricter prompts. Status:

- **ACCEPTED** — all guardrails pass
- **RETRY_LIMIT_REACHED** — failed after 3 attempts
- **PASSED_GUARDRAILS_FAILED** — final attempt failed but no LLM error
- **ERROR** — pipeline error
""",
    )
    if not stage4:
        st.info("Stage 4 didn't run on this set (no HIGH-severity flags, or skipped).")
    else:
        for r in stage4[:20]:
            status_emoji = {
                "ACCEPTED": "✅",
                "RETRY_LIMIT_REACHED": "⚠️",
                "PASSED_GUARDRAILS_FAILED": "⚠️",
                "ERROR": "❌",
            }.get(r.get("status", ""), "❓")
            with st.expander(
                f"{status_emoji} `{r.get('flag_id')}` — **{r.get('status')}** "
                f"(attempts {r.get('attempts')})"
            ):
                col_orig, col_rew = st.columns(2)
                with col_orig:
                    st.markdown("**Original:**")
                    st.text(r.get("original_text", "")[:1000])
                with col_rew:
                    st.markdown("**Rewrite:**")
                    st.text(r.get("rewrite_text", "")[:1000])

                # Guardrail status row
                g1 = r.get("citation_integrity_passed")
                g2 = r.get("reflag_check_passed")
                g3 = r.get("intent_check_passed")
                guardrails = [
                    ("G1 citations", g1),
                    ("G2 BCT recheck", g2),
                    ("G3 intent", g3),
                ]
                cols_g = st.columns(3)
                for col, (name, passed) in zip(cols_g, guardrails):
                    if passed is True:
                        col.success(f"{name}: pass")
                    elif passed is False:
                        col.error(f"{name}: fail")
                    else:
                        col.info(f"{name}: skipped")

                if r.get("rewrite_cd_count") is not None:
                    st.caption(
                        f"BCT non-regression: original CD count = "
                        f"{r.get('original_cd_count', '?')}, "
                        f"rewrite CD count = {r.get('rewrite_cd_count', '?')}. "
                        f"Intent cosine: {r.get('intent_cosine', 0):.3f}."
                    )

                if r.get("is_code_citations"):
                    st.markdown("**Cited IS-code sections (verified):**")
                    for cit in r["is_code_citations"]:
                        ok = "✓" if cit.get("resolved") else "✗"
                        st.markdown(
                            f"- {ok} {cit.get('code', '?')} "
                            f"{cit.get('version') or ''} §{cit.get('section') or '?'}"
                        )

    st.divider()

    # ─── Evaluation (only for frozen demo with ablation data) ─────────────
    if ablation or metrics:
        st.subheader("Evaluation — what your professor will probably ask about")
        explain(
            "Why we evaluate this way",
            """
There is no annotation budget for vagueness in real tenders — every label
needs a domain expert reading surrounding clauses. Two cheaper substitutes:

1. **Synthetic ground truth** for SYN_001 — we wrote the synthetic tender
   ourselves so we know which clauses are vague-by-design (13 vague + 13
   precise = 26 labelled clauses).
2. **Lexicon baseline** — a 37-pattern regex baseline across 6 vagueness
   categories (subjectivity, effort_pledges, delegated_authority, approximation,
   feasibility_dodge, underspecified_quantifier). The pipeline must beat this
   to be worth its API cost.

We report **precision/recall/F1** + **Cohen's κ** between methods.
""",
        )

        if metrics:
            st.markdown("#### Precision / recall against synthetic ground truth")
            rows = []
            for label_key, label_name in [
                ("lexicon_baseline", "Lexicon baseline"),
                ("bct_pipeline", "BCT pipeline (Stage 1 only)"),
                ("bct_confirmed", "BCT + Stage 3 CONFIRMED"),
            ]:
                if label_key in metrics:
                    pr = metrics[label_key]["precision_recall"]
                    rows.append({
                        "Method": label_name,
                        "TP": pr["tp"], "FP": pr["fp"], "FN": pr["fn"],
                        "Precision": f"{pr['precision']:.3f}",
                        "Recall": f"{pr['recall']:.3f}",
                        "F1": f"{pr['f1']:.3f}",
                    })
            if rows:
                st.dataframe(rows, hide_index=True)

            if "agreement" in metrics and "bct_vs_lexicon" in metrics["agreement"]:
                ag = metrics["agreement"]["bct_vs_lexicon"]
                st.markdown(
                    f"**BCT vs lexicon agreement:** "
                    f"both flagged **{ag['both']}** clauses, BCT-only **{ag['only_a']}**, "
                    f"lexicon-only **{ag['only_b']}**, neither **{ag['neither']}**. "
                    f"Cohen's κ = **{ag['cohen_kappa']:.3f}**."
                )
                explain(
                    "Cohen's κ — what does this number mean?",
                    """
Cohen's κ measures agreement between two raters on the same items, corrected
for chance agreement. Range: −1 to +1.

- **κ = 0** — agreement no better than random.
- **κ ≈ 0.2** — *slight* / *fair* agreement (Landis & Koch 1977).
- **κ ≈ 0.4–0.6** — moderate.
- **κ ≈ 0.8** — substantial / near-perfect.

Our κ ≈ 0.20 means **BCT and the regex lexicon catch genuinely different
clauses** — neither subsumes the other. That's the whole point of running
BCT: it finds structural ambiguity (no quantity / method / standard
reference) that a regex matcher cannot.
""",
                )

        if ablation:
            st.markdown("#### Pipeline ablation — precision / recall by stage")
            ablation_rows = []
            for r in ablation:
                ablation_rows.append({
                    "Stage": r["rung"],
                    "Pred+": r["predicted_positive"],
                    "TP": r["tp"], "FP": r["fp"], "FN": r["fn"],
                    "Precision": f"{r['precision']:.3f}",
                    "Recall": f"{r['recall']:.3f}",
                    "F1": f"{r['f1']:.3f}",
                })
            st.dataframe(ablation_rows, hide_index=True)
            explain(
                "How to read the ablation table",
                """
Five filtering rungs, each more selective than the last:

- **A0 No-filter** — flag every clause (gives the universe rate; baseline
  precision ≈ vague rate).
- **A1 Lexicon** — 37-regex baseline.
- **A2 BCT only** — Stage 1 alone.
- **A3 + Stage 2** — keep only flags Stage 2 marked **REAL**.
- **A4 + Stage 3** — keep only flags Stage 3 marked **CONFIRMED**.

The **expected pattern** is precision climbing A2 → A4 while recall holds or
drops slightly. That curve quantifies how much each stage is contributing
to **false-positive suppression** — the core argument for having Stages 2
and 3 at all.

On SYN_001 specifically: precision climbs from 0.611 → 0.636 → 1.000 across
A2 → A4, while recall drops from 0.846 → 0.538 → 0.231. **A4 (full
pipeline) achieves 100% precision** — every flag survived to Stage 3 is a
genuine vague clause.

The lexicon (A1) wins F1 on this synthetic test because SYN_001 was
constructed using explicit lexical markers — *that's* what the lexicon
detects. On real tenders, vagueness is often *structural* (missing
quantity / method / standard with no lexical marker), where BCT outperforms.
""",
            )


# ═══════════════════════════════════════════════════════════════════════════
# DECIDE WHAT TO RENDER
# ═══════════════════════════════════════════════════════════════════════════
if is_live_mode and st.session_state.current_run is not None:
    run = st.session_state.current_run
    art = run["artefacts"]
    render_results(
        title=f"📤 Live result — {run['doc_id']} pp.{run['page_start']}–{run['page_end']}",
        candidates=art["candidates"],
        stage2=art["stage2"],
        stage3=art["stage3"],
        stage4=art["stage4"],
        ablation=None,         # not computed for arbitrary user uploads
        metrics=None,
        summary=run["summary"],
    )
elif not is_live_mode:
    demo = load_demo_artefacts()
    render_results(
        title="📂 Frozen demo — SYN_001 (synthetic tender)",
        candidates=demo["candidates"],
        stage2=demo["stage2"],
        stage3=demo["stage3"],
        stage4=demo["stage4"],
        ablation=demo["ablation"],
        metrics=demo["metrics"],
        summary={"total_cost_inr": 0.0},   # frozen — no live cost
    )
else:
    # Live mode but no run yet
    st.info("👈 Upload a tender package and click **Run full pipeline** in the sidebar to start.")


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Tender Ambiguity Detector — M.Tech thesis, Construction Technology and Management, "
    "Department of Civil Engineering, IIT Bombay (2026). "
    "[GitHub repository](https://github.com/pm-official/mtp-tender-ambiguity-detector)"
)
