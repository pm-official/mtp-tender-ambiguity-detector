"""Stage 2 — apparent-vs-real ambiguity adjudication.

For every CandidateFlag from Stage 1 (BCT), the critic:
  1. Receives the flagged clause text and the BCT findings.
  2. Receives the top-K retrieved chunks via hybrid retrieval (R1+R2+R3+RRF).
  3. Decides REAL / APPARENT / WEAK with rationale and confidence.

Output schema is enforced via Gemini's response_schema. The verdict feeds
the Stage-3 critique pass (next session) and ultimately the user-facing
report.

Default model is `gemini-2.5-flash-lite` (more permissive free-tier quota).
For thesis-grade results, switch to `gemini-2.5-pro` via `--model`.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.llm.gemini_client import LLMCallStats, get_default_client
from src.schemas.bct import CandidateFlag
from src.schemas.stage2 import HybridHit, Stage2Verdict

logger = logging.getLogger(__name__)


STAGE2_SYSTEM_PROMPT = (
    "You are a senior construction-contract reviewer. A clause has been flagged "
    "as potentially vague (specific bidder commitments could not be determined "
    "from the clause alone). Your job is to decide whether the wider tender "
    "package actually resolves the vagueness, or whether it remains real."
)


STAGE2_USER_TEMPLATE = """\
FLAGGED CLAUSE:
  clause_id: {clause_id}
  doc:       {doc_id}  page {page}
  section:   {section}
  text:      <<<{clause_text}>>>

BCT FINDINGS — what the bidder could not determine:
{bct_findings}

RETRIEVED CONTEXT (top-{n_hits}, fused via R1 dense + R2 defined-term + R3 referent):
{retrieved_block}

QUESTIONS:
  Q1  Does any retrieved chunk DEFINE / QUANTIFY / SPECIFY what the BCT said
      was missing (i.e., does it fill in the missing quantity, method, or
      standard)?
  Q2  Does any retrieved chunk EXPLICITLY RESOLVE the vagueness — by
      introducing a measurable threshold, a procedure, or a standard reference
      that applies to this very clause?
  Q3  Does any retrieved chunk CONFLICT with the flagged clause, deepening
      rather than resolving the vagueness?

Decide:
  REAL     — even with the wider context, the vagueness stands; bidder cannot
             commit without further information from the Authority.
  APPARENT — the wider context resolves at least one of the missing fields,
             so the apparent vagueness is in fact pinned down elsewhere.
  WEAK     — borderline; the retrieved context partially resolves but
             material ambiguity remains. Use sparingly.

Be strict. If no retrieved chunk explicitly addresses the missing fields,
verdict is REAL.

If you choose APPARENT or WEAK, return resolving_clause_id pointing to the
single most relevant retrieved clause_id. For REAL, return null.
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
        for q in c.missing_info[:4]:
            lines.append(f"    bidder_q: {q}")
    return "\n".join(lines)


def _format_retrieved_block(hits: list[HybridHit], max_chars: int = 600) -> str:
    if not hits:
        return "  (no clauses retrieved)"
    lines: list[str] = []
    for h in hits:
        meta = h.metadata or {}
        page = meta.get("page", "?")
        section = meta.get("section", "")
        clause_num = meta.get("clause_number") or "-"
        text = h.text.replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        sources = "+".join(h.sources) if h.sources else "?"
        lines.append(
            f"  [{h.rank:>2d}] {h.clause_id}  page {page}  clause {clause_num}  "
            f"section={section[:40]}  routes={sources}  rrf={h.rrf_score:.4f}"
        )
        lines.append(f"       {text}")
    return "\n".join(lines)


def run_stage2_critic(
    flag: CandidateFlag,
    hits: list[HybridHit],
    *,
    model: Optional[str] = None,
) -> tuple[Stage2Verdict, LLMCallStats]:
    """Run the Stage-2 critic on one flag with its retrieved hits.

    Returns (Stage2Verdict, telemetry stats).
    """
    client = get_default_client()
    section = " > ".join(flag.section_path) if flag.section_path else "(none)"
    user_msg = STAGE2_USER_TEMPLATE.format(
        clause_id=flag.clause_id,
        doc_id=flag.doc_id,
        page=flag.page,
        section=section,
        clause_text=flag.clause_text.strip(),
        bct_findings=_format_bct_findings(flag),
        n_hits=len(hits),
        retrieved_block=_format_retrieved_block(hits),
    )

    # Stage 2 is binary classification (REAL vs APPARENT) over retrieved
    # evidence. Flash is sufficient and ~4x cheaper than Pro. Stage 3 still
    # uses Pro for the higher-stakes severity-scored critique. Pass
    # --model gemini-2.5-pro to upgrade if you suspect Flash misclassifies.
    parsed, stats = client.generate(
        prompt=user_msg,
        response_schema=Stage2Verdict,
        model=model or client.default_flash,
        temperature=0.0,
        system_instruction=STAGE2_SYSTEM_PROMPT,
    )

    # Normalise: the LLM-emitted Stage2Verdict may not include flag_id /
    # clause_id (the prompt doesn't ask for them). Patch them in.
    parsed.flag_id = flag.flag_id
    parsed.clause_id = flag.clause_id
    parsed.retrieved_chunk_ids = [h.clause_id for h in hits]
    parsed.retrieved_sources = [h.sources for h in hits]
    parsed.cost_inr = stats.cost_inr

    # Defensive: if the LLM returned an unknown verdict literal, coerce to REAL
    # (conservative — keep the flag rather than dismiss it without evidence).
    if parsed.verdict not in ("REAL", "APPARENT", "WEAK"):
        logger.warning("Unknown verdict %r from LLM; coercing to REAL", parsed.verdict)
        parsed.verdict = "REAL"

    return parsed, stats


__all__ = [
    "run_stage2_critic",
    "STAGE2_SYSTEM_PROMPT",
    "STAGE2_USER_TEMPLATE",
]
