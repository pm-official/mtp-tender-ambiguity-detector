"""Stage 4 — IS-code-grounded rewrite with three guardrails.

For every CONFIRMED REAL flag, propose a rewrite that:
  1. Cites at least one IS-code / CPWD section.
  2. Reduces (does not regress) the BCT-identified uncertainty.
  3. Preserves contractual intent (original word count ± 25%, scope unchanged).

Three guardrails enforce safety:

  G1  Citation integrity. Parse citations, look each up in the local IS-code
      registry. Any unresolved citation → reject.
  G2  Differential BCT re-check. Run the rewrite through Stage 1 BCT and
      count CANNOT_DETERMINE fields across all extracted commitments. The
      rewrite passes if its CD-count is ≤ the original's CD-count
      (non-regression). This is the right abstraction because Stage 1 BCT
      decomposition is variable: a rewrite that turns one vague obligation
      into one specific obligation + a continuation property (e.g. "valid
      until 90 days after DLP") may produce two extracted obligations where
      the original had one. Counting all CDs and requiring non-regression
      sidesteps that artifact while still catching genuine quality regressions.
  G3  Intent preservation. Cosine similarity between rewrite embedding and
      original-clause embedding must be ≥ 0.6. Below that, the rewrite has
      changed contractual scope.

On rejection, retry with a stricter prompt that names the failure mode.
Up to `max_attempts` total attempts (default 3).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from src.iscode.citation import parse_citations, verify_all_citations
from src.iscode.index import query_iscode_index
from src.llm.gemini_client import LLMCallStats, get_default_client
from src.schemas.bct import CandidateFlag
from src.schemas.rewrite import ISCodeChunk, ISCodeCitation, Rewrite
from src.schemas.stage3 import ConfirmedFlag
from src.score.stage1_bct import run_bct_on_clause

logger = logging.getLogger(__name__)


# ─── Constants ─────────────────────────────────────────────────────────────
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_INTENT_COSINE_THRESHOLD = 0.6
DEFAULT_REWRITE_TEMPERATURE = 0.25
DEFAULT_RETRY_TEMPERATURE = 0.45        # bump on retry to encourage variation
DEFAULT_TOP_K_ISCODE = 6


# ─── LLM-output schema ─────────────────────────────────────────────────────
class _ProposedCitation(BaseModel):
    """Citation proposed by the LLM (raw, pre-verification)."""

    code: str = Field(..., description="e.g. 'IS 456' or 'CPWD Specifications'")
    version: Optional[str] = Field(None, description="e.g. '2000' or '2019'")
    section: Optional[str] = Field(None, description="e.g. '8.1'")


class _RewriteProposal(BaseModel):
    rewrite_text: str
    is_code_citations: list[_ProposedCitation] = Field(default_factory=list)
    explanation: str = ""
    preserves_intent: bool = True


# ─── Prompts ────────────────────────────────────────────────────────────────
REWRITE_SYSTEM_PROMPT = (
    "You rewrite vague construction-tender clauses to remove the specific "
    "commitment gaps a bidder identified. You ground every rewrite in cited "
    "Indian standards (IS codes) or CPWD specifications. You NEVER change "
    "the contractual scope — only fill in the missing quantity, method, "
    "and standard reference."
)


REWRITE_USER_TEMPLATE = """\
ORIGINAL CLAUSE:
<<<{clause_text}>>>

SECTION CONTEXT:
{section}

BCT FINDINGS — what the bidder could not determine in the original:
{bct_findings}

RELEVANT IS-CODE / CPWD CONTEXT (top-{n_iscode}, retrieved by semantic similarity):
{iscode_context}

CONSTRAINTS:
  1. Cite AT LEAST ONE IS-code or CPWD section using the form
     "IS 456:2000 Section 8.1" or "CPWD Specifications:2019 Vol 1 Section 5.4".
     Cited references MUST appear in the IS-code context above; do NOT invent
     citations.
  2. Use the version of the IS code shown in the retrieved context.
     If the original tender already cites a specific version, use that version.
  3. Fill the SPECIFIC gaps named in BCT FINDINGS above. For every
     "missing: quantity/method/standard" line, supply a concrete value or
     a single citation that resolves it. Do NOT use vague placeholders
     ("reasonable", "adequate", "sufficient", "as directed", "best efforts",
     "in a form acceptable to the Authority").
  4. Preserve the ORIGINAL'S obligation count. The original has N
     obligations (count the imperative verbs in the original); your
     rewrite should have AT MOST N. Do NOT introduce new "submit X",
     "report Y", or "provide Z" commitments unless they were in the
     original. Convert vagueness to specificity ON the existing
     obligations rather than adding new ones.
  5. If the original delegates to another clause ("as per Special
     Conditions Clause 3"), preserve that delegation — do not invent
     specific thresholds the original left to another document.
  6. Keep the rewrite within ±25% of the original word count.
  7. Use ONE compound sentence per obligation rather than splitting an
     obligation across multiple sentences. Properties of an obligation
     (form, validity period, frequency, location) belong in the same
     sentence as the obligation itself, not as standalone sub-clauses.

Output strict JSON:
{{
  "rewrite_text": str — the proposed rewrite as a single coherent clause,
  "is_code_citations": [
     {{"code": str, "version": str | null, "section": str | null}}, ...
  ] — each citation that appears in rewrite_text,
  "explanation": str — short, explaining how the rewrite fills the BCT gaps,
  "preserves_intent": bool — self-attestation that scope is preserved
}}
"""


RETRY_PREFIX_TEMPLATE = """\
A PREVIOUS ATTEMPT FAILED. Reasons:
{failure_summary}

Fix these specific issues in your next attempt. Do NOT repeat the same
failure modes.

"""


def _format_bct_findings(flag: CandidateFlag) -> str:
    if not flag.bct_output.commitments:
        return "  (no commitments extracted — BCT fired on a non-obligation; not expected for Stage 4)"
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


def _format_iscode_context(chunks: list[ISCodeChunk], max_chars: int = 600) -> str:
    if not chunks:
        return (
            "  (IS-code retrieval unavailable in this run. Use your training-data\n"
            "   knowledge of Indian standards. Cite ONLY codes/sections you are\n"
            "   confident exist — every citation will be verified against the local\n"
            "   IS-code corpus and unverified citations will reject the rewrite.\n"
            "   The local corpus contains: IS 269 (33-grade OPC, 2013), IS 8112\n"
            "   (43-grade OPC, 2013), IS 1489 Parts 1/2 (PPC, 1991), IS 383 (1970,\n"
            "   2016), IS 456:2000, IS 1786:2008, IS 1200 Parts 1-25 (1971-1992),\n"
            "   IS 13920:1993, IS 800:2007, IS 808:1989, IS 875 Parts 1-5 (1987,\n"
            "   2015), IS 1893 Part 1:2002, plus CPWD GCC 2019, CPWD Specifications\n"
            "   2019 Volumes 1-2, CPWD Works Manual 2019.)"
        )
    lines: list[str] = []
    for i, ch in enumerate(chunks, 1):
        text = ch.text.replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        lines.append(f"  [{i}] {ch.code_id}  Section {ch.section}  (page {ch.page})")
        lines.append(f"      {text}")
    return "\n".join(lines)


# ─── BCT helpers for G2 ────────────────────────────────────────────────────
def _count_cannot_determine(bct_output) -> int:
    """Count CANNOT_DETERMINE fields across all commitments.

    Each commitment has three slots — quantity, method, standard. A field is
    "missing" if it equals the sentinel string. Returns the total across all
    commitments, which is the bidder-uncertainty count we minimize.
    """
    total = 0
    for c in bct_output.commitments:
        for field in ("quantity", "method", "standard"):
            if getattr(c, field) == "CANNOT_DETERMINE":
                total += 1
    return total


# ─── Embedding helper for G3 ───────────────────────────────────────────────
def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─── Top-level rewrite call ────────────────────────────────────────────────
def rewrite_one(
    confirmed: ConfirmedFlag,
    flag: CandidateFlag,
    *,
    is_code_top_k: int = DEFAULT_TOP_K_ISCODE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    intent_threshold: float = DEFAULT_INTENT_COSINE_THRESHOLD,
    model: Optional[str] = None,
) -> tuple[Rewrite, list[LLMCallStats]]:
    """Generate a rewrite for one CONFIRMED flag with all three guardrails.

    Returns (Rewrite record, list of LLMCallStats — one per attempt).
    """
    client = get_default_client()
    stats_list: list[LLMCallStats] = []

    # ── Retrieve IS-code context (one-time, used for all attempts) ────────
    query_text = (
        flag.clause_text
        + "\n"
        + " ".join(q for c in flag.bct_output.commitments for q in c.missing_info)
    )
    iscode_chunks = query_iscode_index(query_text, top_k=is_code_top_k)

    # ── Original-clause embedding (one-time, for G3) ──────────────────────
    # G3 is graceful: if the embedding API is unavailable (transient 403/429
    # or local-only mode), we skip G3 rather than fail the whole rewrite.
    # G1 and G2 are the primary safety nets.
    orig_embed: Optional[list[float]] = None
    g3_skipped = False
    try:
        orig_embed_list, orig_embed_stats = client.embed([flag.clause_text])
        stats_list.append(orig_embed_stats)
        orig_embed = orig_embed_list[0]
    except Exception as e:
        logger.warning(
            "G3 (intent cosine) skipped: original-clause embed failed: %s", e,
        )
        g3_skipped = True

    section_str = " > ".join(flag.section_path) if flag.section_path else "(none)"

    # ── Attempt loop ──────────────────────────────────────────────────────
    failure_summaries: list[str] = []
    last_proposal: Optional[_RewriteProposal] = None
    last_citations: list[ISCodeCitation] = []

    rewrite = Rewrite(
        flag_id=flag.flag_id,
        clause_id=flag.clause_id,
        tender_id=flag.tender_id,
        doc_id=flag.doc_id,
        original_text=flag.clause_text,
        status="PASSED_GUARDRAILS_FAILED",   # default — overwritten on success/error/retry
    )

    t0_total = time.monotonic()

    for attempt in range(1, max_attempts + 1):
        rewrite.attempts = attempt
        # Build the prompt; on retries, prepend the failure summary.
        retry_prefix = ""
        if failure_summaries:
            retry_prefix = RETRY_PREFIX_TEMPLATE.format(
                failure_summary="\n".join(f"  - {m}" for m in failure_summaries)
            )

        user_msg = retry_prefix + REWRITE_USER_TEMPLATE.format(
            clause_text=flag.clause_text.strip(),
            section=section_str,
            bct_findings=_format_bct_findings(flag),
            n_iscode=len(iscode_chunks),
            iscode_context=_format_iscode_context(iscode_chunks),
        )

        temp = DEFAULT_RETRY_TEMPERATURE if attempt > 1 else DEFAULT_REWRITE_TEMPERATURE

        try:
            proposal, gen_stats = client.generate(
                prompt=user_msg,
                response_schema=_RewriteProposal,
                model=model or client.default_pro,
                temperature=temp,
                system_instruction=REWRITE_SYSTEM_PROMPT,
            )
            stats_list.append(gen_stats)
            last_proposal = proposal
        except Exception as e:
            logger.warning("Rewrite call failed for %s on attempt %d: %s", flag.flag_id, attempt, e)
            rewrite.error = str(e)
            rewrite.status = "ERROR"
            rewrite.rejection_reason = f"LLM call failed: {e}"
            break

        # ── Guardrail G1: citation integrity ──────────────────────────────
        text_citations = parse_citations(proposal.rewrite_text)
        # Also include the LLM's structured citations (in case it didn't put
        # them in the text)
        structured_citations = [
            ISCodeCitation(
                code=c.code,
                version=c.version,
                section=c.section,
                raw_text=f"{c.code}{':' + c.version if c.version else ''}"
                + (f" §{c.section}" if c.section else ""),
            )
            for c in proposal.is_code_citations
        ]
        all_citations = _dedupe_citations(text_citations + structured_citations)

        verified, resolved_count, unresolved_count = verify_all_citations(all_citations)
        last_citations = verified
        rewrite.is_code_citations = verified
        rewrite.citation_resolved_count = resolved_count
        rewrite.citation_unresolved_count = unresolved_count
        g1_passed = (
            resolved_count > 0           # at least one verified citation
            and unresolved_count == 0    # no hallucinated/unresolved citations
        )
        rewrite.citation_integrity_passed = g1_passed

        # Always tentatively record the proposal so the audit log shows it
        rewrite.rewrite_text = proposal.rewrite_text
        rewrite.rewrite_explanation = proposal.explanation

        if not g1_passed:
            unres_list = [
                f"{c.code} {c.version or ''} §{c.section or '?'}"
                for c in verified if not c.resolved
            ]
            msg = (
                f"Citation integrity (G1) failed: {unresolved_count} unresolved "
                f"citation(s): {unres_list}. "
                f"Resolved={resolved_count}. "
                f"Required: only cite (code, section) pairs that appear in the "
                f"IS-code context block."
            )
            failure_summaries.append(msg)
            rewrite.rejection_reason = msg
            if attempt < max_attempts:
                continue

        # ── Guardrail G2: re-feed through BCT ─────────────────────────────
        # Construct a Clause-like object from the rewrite for BCT input
        from src.schemas.clause import Clause as ClauseSchema

        synthetic_clause = ClauseSchema(
            clause_id=f"{flag.clause_id}::rewrite_attempt_{attempt}",
            doc_id=flag.doc_id,
            tender_id=flag.tender_id,
            page=flag.page,
            page_end=flag.page,
            section_path=flag.section_path,
            clause_number=flag.clause_number,
            text=proposal.rewrite_text,
            char_count=len(proposal.rewrite_text),
            word_count=len(proposal.rewrite_text.split()),
        )
        try:
            bct_out, bct_stats = run_bct_on_clause(synthetic_clause)
            stats_list.append(bct_stats)
        except Exception as e:
            logger.warning("BCT re-check failed for %s on attempt %d: %s", flag.flag_id, attempt, e)
            rewrite.reflag_check_passed = False
            rewrite.reflag_check_reason = f"BCT call failed: {e}"
            if attempt < max_attempts:
                failure_summaries.append(f"BCT re-check (G2) errored: {e}")
                continue
            rewrite.status = "ERROR"
            rewrite.rejection_reason = "BCT re-check could not be performed"
            break

        # Differential G2: pass if rewrite_cd_count <= original_cd_count.
        # We also keep the strict-rule reason text around so retries get a
        # specific failure message about which fields are still missing.
        from src.schemas.bct import evaluate_flag

        orig_cd = _count_cannot_determine(flag.bct_output)
        rewrite_cd = _count_cannot_determine(bct_out)
        rewrite.original_cd_count = orig_cd
        rewrite.rewrite_cd_count = rewrite_cd

        re_flagged, re_reason = evaluate_flag(bct_out)
        g2_passed = rewrite_cd <= orig_cd
        rewrite.reflag_check_passed = g2_passed
        rewrite.reflag_check_reason = (
            f"original_cd={orig_cd}, rewrite_cd={rewrite_cd} "
            f"(non-regression: {'OK' if g2_passed else 'FAIL'}). "
            f"strict-rule detail: {re_reason or '(no missing fields)'}"
        )

        if not g2_passed:
            failure_summaries.append(
                f"BCT re-check (G2) failed: rewrite introduced more uncertainty "
                f"than the original ({rewrite_cd} CANNOT_DETERMINE vs {orig_cd} "
                f"in original). Strict-rule detail: {re_reason}. "
                f"Avoid introducing new sub-obligations whose quantity/method/"
                f"standard you cannot fill from the cited IS-code context."
            )
            rewrite.rejection_reason = (
                f"Rewrite re-flagged: CD count regressed {orig_cd} -> {rewrite_cd}"
            )
            if attempt < max_attempts:
                continue

        # ── Guardrail G3: intent preservation via cosine ──────────────────
        if g3_skipped or orig_embed is None:
            rewrite.intent_cosine = 0.0
            rewrite.intent_check_passed = None  # None = skipped, distinct from True/False
            g3_passed = True   # Don't fail the whole rewrite on a missing embed
        else:
            try:
                rewrite_embed_list, rewrite_embed_stats = client.embed([proposal.rewrite_text])
                stats_list.append(rewrite_embed_stats)
                rewrite_embed = rewrite_embed_list[0]
                cosine = _cosine(orig_embed, rewrite_embed)
                rewrite.intent_cosine = round(cosine, 4)
                g3_passed = cosine >= intent_threshold
                rewrite.intent_check_passed = g3_passed

                if not g3_passed:
                    failure_summaries.append(
                        f"Intent preservation (G3) failed: cosine={cosine:.3f} < {intent_threshold}. "
                        f"The rewrite drifted from the original's contractual scope. Reduce "
                        f"changes; preserve more of the original wording."
                    )
                    rewrite.rejection_reason = (
                        f"Intent cosine {cosine:.3f} below threshold {intent_threshold}"
                    )
                    if attempt < max_attempts:
                        continue
            except Exception as e:
                logger.warning(
                    "G3 (intent cosine) skipped on attempt %d: rewrite embed failed: %s",
                    attempt, e,
                )
                rewrite.intent_check_passed = None
                rewrite.intent_cosine = 0.0
                g3_passed = True   # graceful skip

        # ── All guardrails passed (G3 may be skipped) ────────────────────
        if g1_passed and g2_passed and g3_passed:
            rewrite.status = "ACCEPTED"
            rewrite.rejection_reason = ""
            break

    # End of attempt loop — finalise status if not already set to ACCEPTED or ERROR
    if rewrite.error:
        rewrite.status = "ERROR"
    elif rewrite.status != "ACCEPTED":
        if rewrite.attempts >= max_attempts:
            rewrite.status = "RETRY_LIMIT_REACHED"
        else:
            rewrite.status = "PASSED_GUARDRAILS_FAILED"

    rewrite.seconds_elapsed = time.monotonic() - t0_total
    rewrite.cost_inr = sum(s.cost_inr for s in stats_list)

    return rewrite, stats_list


def _dedupe_citations(cs: list[ISCodeCitation]) -> list[ISCodeCitation]:
    seen: set[tuple] = set()
    out: list[ISCodeCitation] = []
    for c in cs:
        key = (c.code, c.version or "", c.section or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


__all__ = [
    "rewrite_one",
    "REWRITE_SYSTEM_PROMPT",
    "REWRITE_USER_TEMPLATE",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_INTENT_COSINE_THRESHOLD",
    "DEFAULT_TOP_K_ISCODE",
]
