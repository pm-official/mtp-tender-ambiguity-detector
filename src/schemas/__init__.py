"""Pydantic schemas shared across pipeline stages."""

from src.schemas.bct import (
    CANNOT_DETERMINE,
    CONTRACT_DEFINED,
    BCTOutput,
    CandidateFlag,
    Commitment,
    ObligationCheck,
    ObligationCheckBatch,
    evaluate_flag,
)
from src.schemas.clause import BBox, Clause, DocumentTree, PageInfo
from src.schemas.stage2 import VERDICT_VALUES, HybridHit, Stage2Verdict, Verdict
from src.schemas.rewrite import ISCodeChunk, ISCodeCitation, Rewrite, RewriteStatus
from src.schemas.stage3 import (
    ConfirmedFlag,
    CritiqueVerdict,
    CritiqueVerdictLiteral,
    SeverityScore,
    severity_tier,
)
from src.schemas.structures import QUANTITY_KINDS, Definition, Quantity, Reference

__all__ = [
    # clause
    "BBox", "Clause", "DocumentTree", "PageInfo",
    # extract structures
    "Definition", "Quantity", "Reference", "QUANTITY_KINDS",
    # bct
    "CANNOT_DETERMINE", "CONTRACT_DEFINED",
    "BCTOutput", "CandidateFlag", "Commitment",
    "ObligationCheck", "ObligationCheckBatch", "evaluate_flag",
    # stage 2
    "HybridHit", "Stage2Verdict", "Verdict", "VERDICT_VALUES",
    # stage 3
    "ConfirmedFlag", "CritiqueVerdict", "CritiqueVerdictLiteral",
    "SeverityScore", "severity_tier",
    # stage 4 / rewrite
    "ISCodeChunk", "ISCodeCitation", "Rewrite", "RewriteStatus",
]
