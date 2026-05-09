"""HTML report renderer for the user-facing pipeline output.

Produces a single self-contained HTML file (no external CSS / JS) that a
bidder can open in a browser to review the tender's flagged clauses,
verdicts, and bidder pre-bid questions.

The report is also designed to be the demo artefact for thesis defence:
clean, deterministic, every flag traceable back to its clause, with the
full BCT output shown so the methodology's commitment-elicitation framing
is on display.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.audit.run_manifest import list_runs
from src.critic.pipeline import load_verdicts
from src.critic.stage3_pipeline import load_confirmed
from src.score.pipeline import load_flags


# ─── Styles (inlined for self-containment) ──────────────────────────────────
_CSS = """
  :root { --real:#c0392b; --apparent:#2980b9; --weak:#d35400; --ok:#27ae60; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; padding: 2rem; max-width: 1080px; margin-left: auto; margin-right: auto;
         color: #2c3e50; background: #ecf0f1; }
  h1 { font-size: 1.8rem; margin-top: 0; }
  h2 { font-size: 1.3rem; margin-top: 2rem; border-bottom: 2px solid #bdc3c7; padding-bottom: 0.4rem; }
  .meta { color: #7f8c8d; font-size: 0.9rem; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
  .summary-card { background: #fff; padding: 1rem; border-left: 4px solid #34495e; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .summary-card .num { font-size: 2rem; font-weight: bold; color: #2c3e50; }
  .summary-card .label { font-size: 0.85rem; color: #7f8c8d; margin-top: 0.2rem; }
  .summary-card.real { border-left-color: var(--real); }
  .summary-card.apparent { border-left-color: var(--apparent); }
  .summary-card.weak { border-left-color: var(--weak); }
  .summary-card.high { border-left-color: #c0392b; }
  .summary-card.medium { border-left-color: #f39c12; }
  .summary-card.low { border-left-color: #27ae60; }
  .verdict-pill { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: bold; color: white; text-transform: uppercase; letter-spacing: 0.05em; }
  .verdict-pill.real { background: var(--real); }
  .verdict-pill.apparent { background: var(--apparent); }
  .verdict-pill.weak { background: var(--weak); }
  .verdict-pill.confirmed { background: #8e44ad; }
  .verdict-pill.rejected { background: #16a085; }
  .severity-bar { display: inline-flex; gap: 2px; margin-left: 0.5rem; vertical-align: middle; }
  .severity-bar .seg { width: 16px; height: 16px; border-radius: 2px; background: #ecf0f1; border: 1px solid #bdc3c7; }
  .severity-bar .seg.filled.high { background: #c0392b; border-color: #c0392b; }
  .severity-bar .seg.filled.medium { background: #f39c12; border-color: #f39c12; }
  .severity-bar .seg.filled.low { background: #27ae60; border-color: #27ae60; }
  .severity-pill { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: bold; color: white; margin-left: 0.4rem; }
  .severity-pill.high { background: #c0392b; }
  .severity-pill.medium { background: #f39c12; }
  .severity-pill.low { background: #27ae60; }
  .stage3 { background: #f0e6f4; padding: 0.8rem 1rem; margin-top: 0.7rem; border-radius: 4px; border-left: 3px solid #8e44ad; font-size: 0.92rem; }
  .stage3.rejected { border-left-color: #16a085; background: #e6f4ee; }
  .stage3.weak { border-left-color: var(--weak); background: #fdf2e9; }
  .flag { background: #fff; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.07); padding: 1.2rem 1.5rem; margin: 1rem 0; }
  .flag .head { display: flex; gap: 0.7rem; align-items: center; margin-bottom: 0.7rem; flex-wrap: wrap; }
  .flag .head .meta { color: #7f8c8d; font-size: 0.85rem; font-family: ui-monospace, "Courier New", monospace; }
  .clause-text { background: #f8f9fa; border-left: 3px solid #95a5a6; padding: 0.7rem 1rem; margin: 0.5rem 0;
                 font-style: italic; color: #34495e; line-height: 1.5; }
  .commitment { background: #fdfdfd; border: 1px solid #e0e0e0; border-radius: 4px; padding: 0.7rem 1rem; margin: 0.5rem 0; }
  .commitment h4 { margin: 0 0 0.4rem 0; font-size: 0.95rem; }
  .commitment .field { display: flex; gap: 0.4rem; align-items: baseline; font-size: 0.9rem; line-height: 1.5; }
  .commitment .field .key { font-weight: bold; min-width: 80px; color: #34495e; }
  .commitment .field .val.cannot { color: #c0392b; font-weight: bold; font-family: ui-monospace, "Courier New", monospace; }
  .commitment .field .val.contract { color: #16a085; font-family: ui-monospace, "Courier New", monospace; }
  .commitment .field .val.set { color: #27ae60; }
  .missing-info { background: #fff7e6; border-left: 3px solid #f39c12; padding: 0.5rem 1rem; margin-top: 0.5rem; }
  .missing-info ul { margin: 0.3rem 0 0 0; padding-left: 1.5rem; }
  .missing-info li { margin: 0.2rem 0; }
  .stage2 { background: #f5f7fa; padding: 0.8rem 1rem; margin-top: 0.7rem; border-radius: 4px; border-left: 3px solid var(--apparent); font-size: 0.92rem; }
  .stage2.real { border-left-color: var(--real); }
  .stage2.weak { border-left-color: var(--weak); }
  .stage2 .rationale { color: #34495e; margin-top: 0.4rem; line-height: 1.5; }
  .stage2 .resolved { color: #2980b9; font-family: ui-monospace, "Courier New", monospace; }
  .pre-bid-letter { background: #f4f9f4; border: 1px solid #c8e6c9; padding: 1rem 1.5rem; margin: 1.5rem 0; border-radius: 6px; }
  .pre-bid-letter ol { padding-left: 1.5rem; }
  .pre-bid-letter li { margin: 0.4rem 0; }
  footer { color: #95a5a6; font-size: 0.8rem; text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #bdc3c7; }
"""


# ─── Helpers ────────────────────────────────────────────────────────────────
def _sentinel_class(value: str) -> str:
    if value == "CANNOT_DETERMINE":
        return "cannot"
    if value == "CONTRACT_DEFINED":
        return "contract"
    return "set"


def _esc(s: object) -> str:
    return html.escape("" if s is None else str(s))


# ─── Top-level render ──────────────────────────────────────────────────────
def render_html_report(tender_dir: Path, *, out_path: Optional[Path] = None) -> Path:
    """Render the full pipeline output as a self-contained HTML file."""
    tender_dir = Path(tender_dir)
    flags = load_flags(tender_dir)
    verdicts = load_verdicts(tender_dir)
    confirmed = load_confirmed(tender_dir)
    runs = list_runs(tender_dir)

    flagged = [f for f in flags if f.flagged]
    verdict_by_flag = {v.flag_id: v for v in verdicts}
    confirmed_by_flag = {c.flag_id: c for c in confirmed}

    real_flags = [f for f in flagged if (v := verdict_by_flag.get(f.flag_id)) and v.verdict == "REAL" and not v.error]
    apparent_flags = [f for f in flagged if (v := verdict_by_flag.get(f.flag_id)) and v.verdict == "APPARENT"]
    weak_flags = [f for f in flagged if (v := verdict_by_flag.get(f.flag_id)) and v.verdict == "WEAK"]

    # Stage 3 outcomes (only set if Stage 3 has run)
    s3_confirmed = [c for c in confirmed if c.critique_verdict == "CONFIRMED"]
    s3_rejected  = [c for c in confirmed if c.critique_verdict == "REJECTED"]
    s3_weak      = [c for c in confirmed if c.critique_verdict == "WEAK"]
    s3_severity_high   = [c for c in s3_confirmed if c.severity_tier == "high"]
    s3_severity_medium = [c for c in s3_confirmed if c.severity_tier == "medium"]
    s3_severity_low    = [c for c in s3_confirmed if c.severity_tier == "low"]

    # Aggregate all bidder questions — prefer s3 CONFIRMED if Stage 3 ran,
    # otherwise fall back to Stage 2 REAL flags.
    if s3_confirmed:
        question_source = [(c.clause_id, " > ".join(c.section_path or []), q,
                            c.composite_severity, c.severity_tier)
                           for c in sorted(s3_confirmed, key=lambda x: -x.composite_severity)
                           for f in [next((f for f in real_flags if f.flag_id == c.flag_id), None)] if f
                           for com in f.bct_output.commitments
                           for q in com.missing_info]
    else:
        question_source = [(f.clause_id, " > ".join(f.section_path or []), q, 0, "")
                           for f in real_flags
                           for c in f.bct_output.commitments
                           for q in c.missing_info]
    bidder_questions = question_source

    # Total cost across runs
    total_cost_inr = sum(r.get("cost_inr", 0) or 0 for r in runs)
    total_calls = sum(r.get("api_calls", 0) or 0 for r in runs)

    parts: list[str] = []
    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    parts.append(f"<title>Tender Ambiguity Report — {_esc(tender_dir.name)}</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head><body>")

    # ── Header ─────────────────────────────────────────────────────────────
    parts.append(f"<h1>Tender Ambiguity Report</h1>")
    parts.append(
        f"<div class='meta'>tender_id <code>{_esc(tender_dir.name)}</code> &middot; "
        f"generated {_esc(datetime.now(timezone.utc).isoformat(timespec='seconds'))}</div>"
    )

    # ── Summary cards ──────────────────────────────────────────────────────
    parts.append("<div class='summary-grid'>")
    parts.append(_card(len(flags), "Clauses analysed", ""))
    parts.append(_card(len(flagged), "Stage 1 (BCT) flagged", ""))
    parts.append(_card(len(real_flags), "Stage 2: REAL", "real"))
    parts.append(_card(len(apparent_flags), "Stage 2: APPARENT", "apparent"))
    parts.append(_card(len(weak_flags), "Stage 2: WEAK", "weak"))
    if confirmed:
        parts.append(_card(len(s3_confirmed), "Stage 3: CONFIRMED", "high"))
        parts.append(_card(len(s3_rejected), "Stage 3: REJECTED", "low"))
        parts.append(_card(len(s3_weak), "Stage 3: WEAK", "weak"))
    parts.append(_card(len(bidder_questions), "Pre-bid questions extracted", ""))
    parts.append("</div>")

    # ── Severity tier overview if Stage 3 ran ─────────────────────────────
    if s3_confirmed:
        parts.append("<div class='summary-grid'>")
        parts.append(_card(len(s3_severity_high),   "High severity flags",   "high"))
        parts.append(_card(len(s3_severity_medium), "Medium severity flags", "medium"))
        parts.append(_card(len(s3_severity_low),    "Low severity flags",    "low"))
        parts.append("</div>")

    parts.append(
        f"<div class='meta'>Total Gemini API calls across all stages: {total_calls} "
        f"&middot; total cost: INR {total_cost_inr:.4f}</div>"
    )

    # ── Pre-bid clarification letter (the practitioner deliverable) ────────
    if bidder_questions:
        parts.append("<h2>Pre-bid clarification letter (draft) — high-severity first</h2>")
        parts.append(
            "<div class='pre-bid-letter'>"
            "<p>The following questions arise from clauses where the Bidder Commitment Test "
            "could not pin down a specific quantity, method, or standard, and the wider tender "
            "package did not resolve the gap. We respectfully request the Authority's clarification. "
            "Questions are ordered by composite severity (commercial exposure + dispute likelihood).</p>"
            "<ol>"
        )
        seen = set()
        for tup in bidder_questions:
            cid, section, q, sev, tier = tup
            key = q.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            tier_chip = ""
            if tier:
                tier_chip = f" <span class='severity-pill {tier}'>{tier} ({sev}/10)</span>"
            parts.append(
                f"<li><b>Re. clause <code>{_esc(cid)}</code></b>{tier_chip} "
                f"({_esc(section[:80])}): {_esc(q)}</li>"
            )
        parts.append("</ol></div>")

    # ── CONFIRMED flags (Stage 3 output, high → low severity) ──────────────
    if s3_confirmed:
        parts.append("<h2>CONFIRMED ambiguities (post-Stage-3, sorted by severity)</h2>")
        # Sort high → low severity
        s3_confirmed_sorted = sorted(s3_confirmed, key=lambda c: -c.composite_severity)
        for c in s3_confirmed_sorted:
            f = next((f for f in real_flags if f.flag_id == c.flag_id), None)
            if f is None:
                continue
            parts.append(_render_flag(f, verdict_by_flag.get(f.flag_id), c))

    # ── Stage-3 REJECTED flags (Stage 2 said REAL but critique refuted) ────
    if s3_rejected:
        parts.append("<h2>REJECTED on Stage-3 critique pass (Stage 2 was over-strict)</h2>")
        for c in s3_rejected:
            f = next((f for f in real_flags if f.flag_id == c.flag_id), None)
            if f is None:
                continue
            parts.append(_render_flag(f, verdict_by_flag.get(f.flag_id), c))

    # ── REAL flags that haven't gone through Stage 3 yet ───────────────────
    if real_flags and not s3_confirmed and not s3_rejected:
        parts.append("<h2>REAL ambiguities (Stage 2 only — Stage 3 not yet run)</h2>")
        for f in real_flags:
            parts.append(_render_flag(f, verdict_by_flag.get(f.flag_id)))

    # ── APPARENT flags (resolved by retrieval) ─────────────────────────────
    if apparent_flags:
        parts.append("<h2>APPARENT ambiguities (resolved elsewhere in the tender)</h2>")
        for f in apparent_flags:
            parts.append(_render_flag(f, verdict_by_flag.get(f.flag_id)))

    # ── WEAK flags ─────────────────────────────────────────────────────────
    if weak_flags:
        parts.append("<h2>WEAK / borderline cases</h2>")
        for f in weak_flags:
            parts.append(_render_flag(f, verdict_by_flag.get(f.flag_id)))

    # ── Evaluation: ablation + ground-truth metrics (best-effort) ─────────
    parts.append(_render_eval_section(tender_dir))

    # ── Footer ─────────────────────────────────────────────────────────────
    parts.append(
        "<footer>"
        "Generated by the Tender Ambiguity Detector pipeline (Bidder Commitment Test methodology). "
        "Stage 1: per-clause commitment elicitation. Stage 2: hybrid retrieval + apparent-vs-real critic. "
        "Decisions are deterministic from structured LLM output; every flag is auditable in the corpus's "
        "<code>parsed/</code> and <code>runs/</code> directories. "
        "<br/><br/>"
        "M.Tech thesis project — Construction Technology and Management, IIT Bombay, May 2026."
        "</footer>"
    )
    parts.append("</body></html>")

    if out_path is None:
        out_path = tender_dir / "reports" / "report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


def _card(num: int, label: str, css_class: str) -> str:
    cls = f" {css_class}" if css_class else ""
    return f"<div class='summary-card{cls}'><div class='num'>{num}</div><div class='label'>{_esc(label)}</div></div>"


def _render_flag(flag, verdict, confirmed=None) -> str:
    section = " > ".join(flag.section_path or [])
    clause_num = flag.clause_number or "-"
    parts: list[str] = []
    parts.append("<div class='flag'>")
    parts.append("<div class='head'>")
    if verdict:
        v = verdict.verdict.lower()
        parts.append(f"<span class='verdict-pill {v}'>{_esc(verdict.verdict)}</span>")
        if verdict.confidence:
            parts.append(f"<span class='meta'>S2 confidence {verdict.confidence:.2f}</span>")
    if confirmed and confirmed.critique_verdict:
        cv = confirmed.critique_verdict.lower()
        parts.append(f"<span class='verdict-pill {cv}'>{_esc(confirmed.critique_verdict)}</span>")
    if confirmed and confirmed.severity:
        sv = confirmed.composite_severity
        tier = confirmed.severity_tier
        parts.append(f"<span class='severity-pill {tier}'>severity {sv}/10 ({tier})</span>")
    parts.append(f"<span class='meta'>{_esc(flag.doc_id)} &middot; page {flag.page} &middot; clause {_esc(clause_num)}</span>")
    parts.append(f"<span class='meta'>{_esc(section[:80])}</span>")
    parts.append("</div>")

    parts.append(f"<div class='clause-text'>{_esc(flag.clause_text.strip())}</div>")

    # BCT commitments
    for i, c in enumerate(flag.bct_output.commitments, 1):
        parts.append("<div class='commitment'>")
        parts.append(f"<h4>Commitment {i}: {_esc(c.obligation)}</h4>")
        for key, val in (("quantity", c.quantity), ("method", c.method), ("standard", c.standard)):
            cls = _sentinel_class(val)
            parts.append(
                f"<div class='field'><span class='key'>{key}:</span>"
                f"<span class='val {cls}'>{_esc(val)}</span></div>"
            )
        if c.missing_info:
            parts.append("<div class='missing-info'><b>Bidder questions:</b><ul>")
            for q in c.missing_info:
                parts.append(f"<li>{_esc(q)}</li>")
            parts.append("</ul></div>")
        parts.append("</div>")

    # Stage 2 verdict
    if verdict and not verdict.error:
        v = verdict.verdict.lower()
        parts.append(f"<div class='stage2 {v}'>")
        parts.append(f"<b>Stage 2 ({_esc(verdict.verdict)}):</b><br/>")
        if verdict.resolving_clause_id:
            parts.append(f"<b>Resolved by:</b> <span class='resolved'>{_esc(verdict.resolving_clause_id)}</span><br/>")
        if verdict.rationale:
            parts.append(f"<div class='rationale'>{_esc(verdict.rationale)}</div>")
        parts.append("</div>")
    elif verdict and verdict.error:
        parts.append(f"<div class='stage2 real'><b>Stage 2 failed:</b> {_esc(verdict.error[:200])}</div>")

    # Stage 3 verdict + severity
    if confirmed and confirmed.critique_verdict:
        cv = confirmed.critique_verdict.lower()
        parts.append(f"<div class='stage3 {cv}'>")
        parts.append(f"<b>Stage 3 critique ({_esc(confirmed.critique_verdict)}):</b><br/>")
        if confirmed.critique_refutation:
            parts.append(f"<div class='rationale'>{_esc(confirmed.critique_refutation)}</div>")
        if confirmed.severity:
            sv = confirmed.severity
            parts.append(
                f"<br/><b>Severity:</b> commercial {sv.commercial_exposure}/5 &middot; "
                f"dispute {sv.dispute_likelihood}/5 &middot; reviewer cost {sv.reviewer_cost}/5"
            )
            if sv.rationale:
                parts.append(f"<div class='rationale'>{_esc(sv.rationale)}</div>")
        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


# ─── Eval section (Session 11 deliverable) ─────────────────────────────────
def _render_eval_section(tender_dir: Path) -> str:
    """Render the Evaluation section: ablation table + ground-truth metrics
    + lexicon-baseline agreement. Best-effort — if the required JSON files
    are absent, the section is skipped silently."""
    import json as _json

    parsed = tender_dir / "parsed"
    metrics_path = parsed / "metrics_synthetic.json"
    ablation_path = parsed / "ablation.json"
    lex_summary_path = parsed / "baseline_lexicon_summary.json"

    if not (metrics_path.exists() or ablation_path.exists()):
        return ""

    parts: list[str] = ['<h2 id="eval">Evaluation</h2>']
    parts.append(
        "<div class='meta'>The numbers below are computed against a hand-labelled "
        "ground-truth set of vague-by-design clauses (for synthetic tenders) and "
        "the rule-based lexicon baseline. They are reproducible via "
        "<code>python -m src.cli eval ablation --tender-id &lt;id&gt;</code>.</div>"
    )

    # Ground-truth metrics
    if metrics_path.exists():
        try:
            m = _json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            m = None
        if m:
            parts.append(
                "<h3>Precision / recall against synthetic ground truth</h3>"
                "<table style='border-collapse:collapse; margin: 0.5rem 0;'>"
                "<thead><tr>"
                "<th style='text-align:left; padding:4px 12px;'>Method</th>"
                "<th style='text-align:right; padding:4px 12px;'>TP</th>"
                "<th style='text-align:right; padding:4px 12px;'>FP</th>"
                "<th style='text-align:right; padding:4px 12px;'>FN</th>"
                "<th style='text-align:right; padding:4px 12px;'>Precision</th>"
                "<th style='text-align:right; padding:4px 12px;'>Recall</th>"
                "<th style='text-align:right; padding:4px 12px;'>F1</th>"
                "</tr></thead><tbody>"
            )
            rows = []
            if "lexicon_baseline" in m:
                pr = m["lexicon_baseline"]["precision_recall"]
                rows.append(("Lexicon baseline", pr))
            if "bct_pipeline" in m:
                pr = m["bct_pipeline"]["precision_recall"]
                rows.append(("BCT pipeline (Stage 1)", pr))
            if "bct_confirmed" in m:
                pr = m["bct_confirmed"]["precision_recall"]
                rows.append(("BCT + Stage 3 CONFIRMED", pr))
            for name, pr in rows:
                parts.append(
                    f"<tr>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee;'>{_esc(name)}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{pr['tp']}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{pr['fp']}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{pr['fn']}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{pr['precision']:.3f}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{pr['recall']:.3f}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{pr['f1']:.3f}</td>"
                    f"</tr>"
                )
            parts.append("</tbody></table>")

            if "agreement" in m and "bct_vs_lexicon" in m["agreement"]:
                ag = m["agreement"]["bct_vs_lexicon"]
                parts.append(
                    f"<div class='meta'>BCT vs lexicon agreement: "
                    f"both flagged {ag['both']}, BCT-only {ag['only_a']}, "
                    f"lexicon-only {ag['only_b']}, neither {ag['neither']} "
                    f"(Cohen's κ = {ag['cohen_kappa']:.3f}).</div>"
                )

    # Ablation table
    if ablation_path.exists():
        try:
            rows = _json.loads(ablation_path.read_text(encoding="utf-8"))
        except Exception:
            rows = None
        if rows:
            parts.append(
                "<h3>Pipeline ablation — precision/recall by stage</h3>"
                "<table style='border-collapse:collapse; margin: 0.5rem 0;'>"
                "<thead><tr>"
                "<th style='text-align:left; padding:4px 12px;'>Stage</th>"
                "<th style='text-align:right; padding:4px 12px;'>Pred+</th>"
                "<th style='text-align:right; padding:4px 12px;'>TP</th>"
                "<th style='text-align:right; padding:4px 12px;'>FP</th>"
                "<th style='text-align:right; padding:4px 12px;'>FN</th>"
                "<th style='text-align:right; padding:4px 12px;'>Precision</th>"
                "<th style='text-align:right; padding:4px 12px;'>Recall</th>"
                "<th style='text-align:right; padding:4px 12px;'>F1</th>"
                "</tr></thead><tbody>"
            )
            for r in rows:
                parts.append(
                    f"<tr>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee;'><code>{_esc(r['rung'])}</code></td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{r['predicted_positive']}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{r['tp']}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{r['fp']}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{r['fn']}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{r['precision']:.3f}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{r['recall']:.3f}</td>"
                    f"<td style='padding:4px 12px; border-top:1px solid #eee; text-align:right;'>{r['f1']:.3f}</td>"
                    f"</tr>"
                )
            parts.append("</tbody></table>")
            parts.append(
                "<div class='meta'>"
                "A0 = no filter (universe). A1 = lexicon baseline. "
                "A2 = Stage-1 BCT only. A3 = + Stage-2 critic (REAL only). "
                "A4 = + Stage-3 confirm (CONFIRMED only). The expected pattern is "
                "precision climbing A2→A4 while recall drops, which quantifies "
                "each stage's contribution to false-positive suppression."
                "</div>"
            )

    # Lexicon-baseline summary
    if lex_summary_path.exists():
        try:
            ls = _json.loads(lex_summary_path.read_text(encoding="utf-8"))
        except Exception:
            ls = None
        if ls:
            cats = ls.get("category_counts") or {}
            cats_str = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()))
            parts.append(
                f"<div class='meta'>Lexicon baseline: "
                f"{ls.get('clauses_flagged', 0)} of {ls.get('clauses_total', 0)} clauses "
                f"flagged ({100 * ls.get('flag_rate', 0):.1f}%). "
                f"Category breakdown: {_esc(cats_str)}.</div>"
            )

    return "".join(parts)
