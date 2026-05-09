"""Pydantic schemas for Stage 2 (apparent-vs-real ambiguity adjudication).

The hybrid retrieval layer (R1+R2+R3 fused via RRF) produces a list of
HybridHit. The Stage-2 LLM critic then takes the flagged clause, its BCT
output, and these retrieved hits, and emits a Stage2Verdict.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HybridHit(BaseModel):
    """One retrieved candidate from the hybrid retrieval layer."""

    clause_id: str
    text: str = ""
    metadata: dict = Field(default_factory=dict)
    rrf_score: float = 0.0           # reciprocal-rank-fused score
    rank: int = 0                    # final rank in fused list (1-indexed)
    sources: list[str] = Field(default_factory=list)   # routes that returned this hit


# Verdict literals — kept as strings so the prompt and the schema agree.
Verdict = Literal["REAL", "APPARENT", "WEAK"]
VERDICT_VALUES = ("REAL", "APPARENT", "WEAK")


class Stage2Verdict(BaseModel):
    """Stage 2 adjudication outcome for a single CandidateFlag."""

    flag_id: str
    clause_id: str
    verdict: Verdict
    resolving_clause_id: Optional[str] = None     # id of the clause that resolves it (if APPARENT/WEAK)
    rationale: str = ""                           # short LLM-written explanation
    confidence: float = 0.0                       # 0..1, LLM-self-reported

    # Retrieval audit
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_sources: list[list[str]] = Field(default_factory=list)
                          # parallel to retrieved_chunk_ids; lists routes that surfaced each

    # Cost / latency telemetry
    cost_inr: float = 0.0
    seconds_elapsed: float = 0.0
    error: Optional[str] = None
