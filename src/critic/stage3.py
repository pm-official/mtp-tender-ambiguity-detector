"""Stage 3 — critique pass + severity scoring.

Run AFTER Stage 2 produces verdicts. For each Stage-2 REAL verdict:
  1. Critique pass (skeptical-reviewer LLM call) attempts to refute REAL.
     Outputs CONFIRMED / REJECTED / WEAK.
  2. Severity scoring (3-feature 1-5 scale: commercial exposure, dispute
     likelihood, reviewer cost). Composite severity = commercial + dispute.

Two separate LLM calls per surviving flag — kept distinct so the prompts
can be tuned independently and so reviewers can audit each step.

Defaults: Pro at temperature 0.2 for the critique pass (Methodology Part
2.5.3 — diversity helps catch errors). Pro at temperature 0.0 for severity
(deterministic).
"""

from __future__ import annotations

import logging
from typing import Optional

from src.llm.gemini_client import LLMCallStats, get_default_client
from src.schemas.bct import CandidateFlag
from src.schemas.stage2 import HybridHit, Stage2Verdict
from src.schemas.stage3 import CritiqueVerdict, SeverityScore

logger = logging.getLogger(__name__)


# ─── Critique pass ─────────────────────────────────────────────────────────
CRITIQUE_SYSTEM_PROMPT = (
    "You are a skeptical senior construction-contract reviewer with 25 "
    "years of experience. A junior reviewer has flagged the clause below "
    "as REAL ambiguity. Your job is to AGGRESSIVELY try to refute that "
    "verdict — find ANY plausible reading of the wider tender that would "
    "let a competent bidder commit. Only if you cannot refute do you "
    "CONFIRM the verdict."
)


CRITIQUE_USER_TEMPLATE = """\
FLAGGED CLAUSE:
  clause_id: {clause_id}
  doc:       {doc_id}  page {page}
  text:      <<<{clause_text}>>>

BCT FINDINGS — what the original Stage-1 critic said was missing:
{bct_findings}

RETRIEVED CONTEXT (top-{n_hits}, fused via hybrid retrieval):
{retrieved_block}

EARLIER (Stage-2) VERDICT: REAL
EARLIER RATIONALE: {stage2_rationale}

Now refute or confirm. Be skeptical of Stage 2 — it may have missed a
subtle resolving clause, or over-interpreted "missing" when the contract
follows industry-standard implicit conventions.

Output strict JSON:
{{
  "verdict": one of ["CONFIRMED", "REJECTED", "WEAK"],
  "refutation": short text — your reasoning. If REJECTED or WEAK,
                 explain WHICH retrieved chunk or which industry
                 convention resolves the gap. If CONFIRMED, explain
                 why the gap genuinely cannot be filled without
                 Authority clarification.
  "resolving_clause_id": string | null — present if REJECTED/WEAK and
                         a specific retrieved clause supplies the answer.
  "confidence": float in [0, 1].
}}

CONFIRMED: Stage 2's REAL verdict stands; the bidder cannot commit
           without clarification.
REJECTED:  Stage 2 was wrong; a specific retrieved chunk OR an
           explicit industry convention resolves the gap.
WEAK:      Borderline. The gap exists in principle but is small enough
           that a competent bidder could price a contingency.
"""


def _format_bct_findings(flag: CandidateFlag) -> str:
    if not flag.bct_output.commitments:
        return "  (no commitments extracted)"
    lines: list[str] = []
    for i, c in enumerate(flag.bct_output.commitments, 1):
        lines.append(f"  Commitment {i}: {c.obligation}")
        gaps = []
        if c.quantity == "CANNOT_DETERMINE":
            gaps.append("quantity")
        if c.method == "CANNOT_DETERMINE":
            gaps.append("method")
        if c.standard == "CANNOT_DETERMINE":
            gaps.append("standard")
        if gaps:
            lines.append(f"    missing: {', '.join(gaps)}")
        for q in c.missing_info[:3]:
            lines.append(f"    bidder_q: {q}")
    return "\n".join(lines)


def _format_retrieved_block(hits: list[HybridHit], max_chars: int = 500) -> str:
    if not hits:
        return "  (no clauses retrieved)"
    lines: list[str] = []
    for h in hits:
        meta = h.metadata or {}
        page = meta.get("page", "?")
        clause_num = meta.get("clause_number") or "-"
        text = h.text.replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        sources = "+".join(h.sources) if h.sources else "?"
        lines.append(
            f"  [{h.rank:>2d}] {h.clause_id}  page {page} clause {clause_num}  routes={sources}"
        )
        lines.append(f"       {text}")
    return "\n".join(lines)


def run_critique_pass(
    flag: CandidateFlag,
    stage2: Stage2Verdict,
    hits: list[HybridHit],
    *,
    model: Optional[str] = None,
) -> tuple[CritiqueVerdict, LLMCallStats]:
    """Run the critique pass on one Stage-2 REAL verdict.

    Returns (CritiqueVerdict, telemetry stats). The verdict is patched with
    flag_id/clause_id even if the LLM omits them.
    """
    client = get_default_client()
    user_msg = CRITIQUE_USER_TEMPLATE.format(
        clause_id=flag.clause_id,
        doc_id=flag.doc_id,
        page=flag.page,
        clause_text=flag.clause_text.strip(),
        bct_findings=_format_bct_findings(flag),
        n_hits=len(hits),
        retrieved_block=_format_retrieved_block(hits),
        stage2_rationale=stage2.rationale,
    )

    parsed, stats = client.generate(
        prompt=user_msg,
        response_schema=CritiqueVerdict,
        model=model or client.default_pro,
        temperature=0.2,                          # diversity per methodology
        system_instruction=CRITIQUE_SYSTEM_PROMPT,
    )

    parsed.flag_id = flag.flag_id
    parsed.clause_id = flag.clause_id

    if parsed.verdict not in ("CONFIRMED", "REJECTED", "WEAK"):
        logger.warning("Unknown critique verdict %r; coercing to CONFIRMED", parsed.verdict)
        parsed.verdict = "CONFIRMED"

    return parsed, stats


# ─── Severity scoring ──────────────────────────────────────────────────────
SEVERITY_SYSTEM_PROMPT = (
    "You are a contract-risk economist with 20 years of experience pricing "
    "Indian construction tenders. You score the financial and dispute risk "
    "of confirmed-ambiguous clauses on a 1–5 scale per dimension."
)


SEVERITY_USER_TEMPLATE = """\
A clause has been CONFIRMED as ambiguous after a critique pass. Score it
on three dimensions, each as an INTEGER 1..5:

CLAUSE: <<<{clause_text}>>>
SECTION: {section}
BCT FINDINGS — what cannot be determined:
{bct_findings}

Dimensions to score:

  COMMERCIAL EXPOSURE (1..5)
    1 = minor procedural; bidder can absorb without pricing impact
    2 = pricing impact bounded to a single small BoQ item
    3 = material to a single BoQ item, may shift unit rates
    4 = affects multiple BoQ items or a milestone payment
    5 = affects Contract Price or a major milestone payment

  DISPUTE LIKELIHOOD (1..5)
    1 = interpretive practice usually resolves; rarely litigated
    2 = occasionally generates pre-bid queries
    3 = sometimes disputed at variation/measurement stage
    4 = often disputed; arbitration plausible
    5 = routinely litigated (BoQ change provisions, priority-of-documents
        conflicts, indemnity caps)

  REVIEWER COST (1..5)
    1 = trivial pre-bid clarification; one-line answer suffices
    2 = simple Authority confirmation
    3 = requires Engineer / Authority Engineer review
    4 = needs Authority technical committee opinion
    5 = requires substantive redrafting or legal review

Output strict JSON:
{{
  "commercial_exposure": int 1..5,
  "dispute_likelihood": int 1..5,
  "reviewer_cost": int 1..5,
  "rationale": short text explaining the scores
}}
"""


def run_severity_scoring(
    flag: CandidateFlag,
    *,
    model: Optional[str] = None,
) -> tuple[SeverityScore, LLMCallStats]:
    """Run severity scoring on one CONFIRMED flag."""
    client = get_default_client()
    section = " > ".join(flag.section_path) if flag.section_path else "(none)"
    user_msg = SEVERITY_USER_TEMPLATE.format(
        clause_text=flag.clause_text.strip(),
        section=section,
        bct_findings=_format_bct_findings(flag),
    )

    parsed, stats = client.generate(
        prompt=user_msg,
        response_schema=SeverityScore,
        model=model or client.default_pro,
        temperature=0.0,
        system_instruction=SEVERITY_SYSTEM_PROMPT,
    )
    return parsed, stats


__all__ = [
    "run_critique_pass",
    "run_severity_scoring",
    "CRITIQUE_SYSTEM_PROMPT",
    "CRITIQUE_USER_TEMPLATE",
    "SEVERITY_SYSTEM_PROMPT",
    "SEVERITY_USER_TEMPLATE",
]
