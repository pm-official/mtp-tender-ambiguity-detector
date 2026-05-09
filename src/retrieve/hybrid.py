"""Hybrid retrieval: three routes fused via Reciprocal Rank Fusion (RRF).

Per methodology Part 6.1:

  R1 — Dense semantic.    ChromaDB top-K (cosine over Gemini embeddings).
       Best for: vagueness, undefined terms, anaphoric, syntactic.

  R2 — Shared defined-term. Inverted index over the definitions table.
       Returns clauses that mention the same defined terms as the flagged
       clause. Best for: undefined terms (Type L), definitional drift (K).

  R3 — Shared canonical referent.  Group-by over the quantities table.
       Returns clauses that quantify on the same canonical referent.
       Best for: numerical inconsistency (H), priority conflict (I).

For Stage 2 (vagueness focus), R1 is the dominant route, but R2 and R3 catch
cases where the resolving information is in a Definitions section or a BoQ
preamble note. The three routes' rankings are fused via RRF (k=60), the
standard IR rank-aggregation method.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from src.extract.vector_index import query_vector_index
from src.schemas.bct import CandidateFlag
from src.schemas.clause import Clause
from src.schemas.stage2 import HybridHit
from src.schemas.structures import Definition, Quantity

logger = logging.getLogger(__name__)


# ─── R1 — dense semantic ────────────────────────────────────────────────────
def retrieve_dense(
    query_text: str,
    *,
    tender_id: str,
    top_k: int = 8,
    persist_dir: Optional[Path] = None,
) -> list[tuple[str, float, str, dict]]:
    """Return [(clause_id, score, text, metadata)] from ChromaDB for the query."""
    hits = query_vector_index(
        query_text,
        tender_id=tender_id,
        top_k=top_k,
        persist_dir=persist_dir,
    )
    return [(h.clause_id, h.score, h.text, h.metadata or {}) for h in hits]


# ─── R2 — shared defined-term ───────────────────────────────────────────────
def _terms_in_text(text: str, all_terms: Iterable[str]) -> list[str]:
    """Return the subset of `all_terms` that appear in `text` (case-insensitive)."""
    lowered = text.lower()
    matched: list[str] = []
    for t in all_terms:
        if not t:
            continue
        if t.lower() in lowered:
            matched.append(t)
    return matched


def retrieve_shared_defined_terms(
    flagged_clause_text: str,
    *,
    definitions: list[Definition],
    clauses: list[Clause],
    top_k: int = 8,
) -> list[tuple[str, float, str, dict]]:
    """Find clauses that mention the same defined-term as the flagged clause.

    Strategy:
      1. Identify which canonical-form terms (from `definitions`) appear in
         the flagged clause text.
      2. For each such term, score every other clause by how many of these
         terms it contains.
      3. Return the top_k clauses by score.
    """
    if not definitions:
        return []

    canonical_terms = {d.canonical_form for d in definitions if d.canonical_form}
    overlap = _terms_in_text(flagged_clause_text, canonical_terms)
    if not overlap:
        return []

    # Score every other clause by # of overlapping defined-terms
    scores: dict[str, int] = defaultdict(int)
    clause_by_id = {c.clause_id: c for c in clauses}
    for c in clauses:
        for t in overlap:
            if t.lower() in c.text.lower():
                scores[c.clause_id] += 1

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    out: list[tuple[str, float, str, dict]] = []
    for rank, (cid, n) in enumerate(ranked, 1):
        if rank > top_k:
            break
        c = clause_by_id.get(cid)
        if not c:
            continue
        meta = {
            "doc_id": c.doc_id,
            "page": c.page,
            "section": " > ".join(c.section_path),
            "clause_number": c.clause_number or "",
            "n_overlapping_terms": n,
        }
        out.append((cid, float(n), c.text, meta))
    return out


# ─── R3 — shared canonical referent ─────────────────────────────────────────
def retrieve_shared_referents(
    flagged_clause: Clause,
    *,
    quantities: list[Quantity],
    clauses: list[Clause],
    top_k: int = 8,
) -> list[tuple[str, float, str, dict]]:
    """Return clauses that quantify on the same canonical referents as the
    flagged clause does.

    For Type-F (vagueness) flags the flagged clause typically has no
    quantity, so this route often returns nothing. It earns its keep on
    flags where the clause does mention a quantity (e.g. "5%" but the
    referent itself is vague — what is 5% of?)."""
    if not quantities:
        return []

    by_clause = defaultdict(list)
    for q in quantities:
        by_clause[q.clause_id].append(q)

    flagged_referents = {q.referent for q in by_clause.get(flagged_clause.clause_id, []) if q.referent and q.referent != "UNCLASSIFIED"}
    if not flagged_referents:
        return []

    scores: dict[str, int] = defaultdict(int)
    clause_by_id = {c.clause_id: c for c in clauses}
    for cid, qs in by_clause.items():
        if cid == flagged_clause.clause_id:
            continue
        for q in qs:
            if q.referent in flagged_referents:
                scores[cid] += 1

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    out: list[tuple[str, float, str, dict]] = []
    for rank, (cid, n) in enumerate(ranked, 1):
        if rank > top_k:
            break
        c = clause_by_id.get(cid)
        if not c:
            continue
        meta = {
            "doc_id": c.doc_id,
            "page": c.page,
            "section": " > ".join(c.section_path),
            "clause_number": c.clause_number or "",
            "shared_referents_count": n,
        }
        out.append((cid, float(n), c.text, meta))
    return out


# ─── RRF fusion ─────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    route_results: dict[str, list[tuple[str, float, str, dict]]],
    *,
    k: int = 60,
    top_k: int = 12,
) -> list[HybridHit]:
    """Fuse multiple ranked lists into one using RRF.

    `route_results[route_name]` is a list of (clause_id, score, text, metadata)
    in DESCENDING order of relevance for that route.

    RRF score = sum over routes r of 1 / (k + rank_r(c)).
    """
    fused: dict[str, dict] = {}
    for route, hits in route_results.items():
        for rank, (cid, _score, text, meta) in enumerate(hits, 1):
            entry = fused.setdefault(cid, {
                "clause_id": cid,
                "text": text,
                "metadata": dict(meta),
                "rrf_score": 0.0,
                "sources": [],
            })
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["sources"].append(route)
            # Prefer the longest text we've seen for the chunk
            if len(text) > len(entry["text"]):
                entry["text"] = text
                entry["metadata"] = dict(meta)

    ranked = sorted(fused.values(), key=lambda x: -x["rrf_score"])
    out: list[HybridHit] = []
    for i, e in enumerate(ranked[:top_k], 1):
        out.append(HybridHit(
            clause_id=e["clause_id"],
            text=e["text"],
            metadata=e["metadata"],
            rrf_score=round(e["rrf_score"], 6),
            rank=i,
            sources=list(dict.fromkeys(e["sources"])),  # preserve order, dedupe
        ))
    return out


# ─── Top-level: retrieve_for_flag ───────────────────────────────────────────
def retrieve_for_flag(
    flag: CandidateFlag,
    *,
    tender_id: str,
    clauses: list[Clause],
    definitions: list[Definition],
    quantities: list[Quantity],
    top_k_per_route: int = 8,
    top_k_fused: int = 12,
    rrf_k: int = 60,
    persist_dir: Optional[Path] = None,
) -> list[HybridHit]:
    """Run all three retrieval routes for one flag and fuse via RRF."""
    flagged_clause = next((c for c in clauses if c.clause_id == flag.clause_id), None)
    if flagged_clause is None:
        logger.warning("Flag %s references missing clause %s", flag.flag_id, flag.clause_id)
        return []

    # Build the dense-retrieval query: clause text + bidder questions
    missing_qs: list[str] = []
    for c in flag.bct_output.commitments:
        missing_qs.extend(c.missing_info)
    query_text = flag.clause_text + "\n" + "\n".join(missing_qs)

    route_results: dict[str, list[tuple[str, float, str, dict]]] = {}

    # R1 — dense semantic (live, hits ChromaDB)
    try:
        r1 = retrieve_dense(
            query_text,
            tender_id=tender_id,
            top_k=top_k_per_route,
            persist_dir=persist_dir,
        )
        # Drop the flagged clause itself from results
        r1 = [(cid, s, t, m) for cid, s, t, m in r1 if cid != flag.clause_id]
        route_results["R1_dense"] = r1
    except Exception as e:
        logger.warning("R1 dense retrieval failed: %s", e)

    # R2 — shared defined-term
    try:
        r2 = retrieve_shared_defined_terms(
            flagged_clause.text,
            definitions=definitions,
            clauses=[c for c in clauses if c.clause_id != flag.clause_id],
            top_k=top_k_per_route,
        )
        route_results["R2_defined_term"] = r2
    except Exception as e:
        logger.warning("R2 lexical retrieval failed: %s", e)

    # R3 — shared canonical referent
    try:
        r3 = retrieve_shared_referents(
            flagged_clause,
            quantities=quantities,
            clauses=[c for c in clauses if c.clause_id != flag.clause_id],
            top_k=top_k_per_route,
        )
        route_results["R3_referent"] = r3
    except Exception as e:
        logger.warning("R3 structural retrieval failed: %s", e)

    return reciprocal_rank_fusion(route_results, k=rrf_k, top_k=top_k_fused)
