"""Vector index of clauses backed by ChromaDB + Gemini text-embedding-004.

One ChromaDB collection per tender, persisted on disk so re-runs are free.

Embedding strategy:
  * Use Gemini text-embedding-004 via our cached client.
  * Skip clauses already in the collection (idempotent).
  * Hash each clause's text to detect content drift.

Query strategy:
  * Caller passes a query string and gets back top-K (clause_id, score, text)
    tuples plus the clause metadata.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from src.schemas.clause import Clause

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Outcome of build_vector_index()."""

    collection_name: str
    total_clauses_seen: int
    newly_embedded: int
    skipped_already_indexed: int
    total_in_collection: int
    persist_dir: Path
    cost_usd: float = 0.0
    cost_inr: float = 0.0


@dataclass
class QueryHit:
    clause_id: str
    score: float          # cosine similarity, higher = more similar
    text: str
    metadata: dict        # all metadata stored with the doc
    rank: int             # 1-indexed


# ─── Helpers ────────────────────────────────────────────────────────────────
def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _persist_dir_for(tender_id: str) -> Path:
    return _project_root() / ".cache" / "chroma" / tender_id


def _collection_name(tender_id: str) -> str:
    """ChromaDB collection names must be 3-63 chars, [a-zA-Z0-9._-]."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tender_id)
    safe = safe.strip("_.-") or "tender"
    return f"tender_{safe}"[:63]


# ─── Build ──────────────────────────────────────────────────────────────────
def build_vector_index(
    clauses: Iterable[Clause],
    *,
    tender_id: str,
    persist_dir: Optional[Path] = None,
    batch_size: int = 64,
) -> IndexResult:
    """Embed every clause that isn't already in the persistent collection."""
    try:
        import chromadb
    except ImportError as e:
        raise ImportError("chromadb not installed") from e

    from src.llm.gemini_client import get_default_client

    persist = Path(persist_dir or _persist_dir_for(tender_id))
    persist.mkdir(parents=True, exist_ok=True)

    chroma = chromadb.PersistentClient(path=str(persist))
    coll_name = _collection_name(tender_id)
    collection = chroma.get_or_create_collection(name=coll_name)

    existing_ids: set[str] = set(collection.get()["ids"] or [])

    clauses = list(clauses)
    to_embed_clauses: list[Clause] = []
    for c in clauses:
        if c.clause_id in existing_ids:
            continue
        to_embed_clauses.append(c)

    skipped = len(clauses) - len(to_embed_clauses)
    cost_usd = 0.0
    cost_inr = 0.0
    newly_embedded = 0

    if to_embed_clauses:
        client = get_default_client()
        for i in range(0, len(to_embed_clauses), batch_size):
            chunk = to_embed_clauses[i : i + batch_size]
            texts = [c.text for c in chunk]
            embeddings, stats = client.embed(texts)

            cost_usd += stats.cost_usd
            cost_inr += stats.cost_inr

            ids = [c.clause_id for c in chunk]
            metadatas = [_metadata_for(c) for c in chunk]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            newly_embedded += len(chunk)
            logger.info(
                "Embedded batch %d-%d / %d (cumulative cost: USD %.6f)",
                i, i + len(chunk), len(to_embed_clauses), cost_usd,
            )

    total_now = collection.count()
    return IndexResult(
        collection_name=coll_name,
        total_clauses_seen=len(clauses),
        newly_embedded=newly_embedded,
        skipped_already_indexed=skipped,
        total_in_collection=total_now,
        persist_dir=persist,
        cost_usd=cost_usd,
        cost_inr=cost_inr,
    )


def _metadata_for(c: Clause) -> dict:
    return {
        "doc_id": c.doc_id,
        "tender_id": c.tender_id,
        "page": c.page,
        "section": " > ".join(c.section_path),
        "clause_number": c.clause_number or "",
        "table_origin": bool(c.table_origin),
        "char_count": c.char_count,
        "content_hash": _content_hash(c.text),
    }


# ─── Query ──────────────────────────────────────────────────────────────────
def query_vector_index(
    query_text: str,
    *,
    tender_id: str,
    top_k: int = 8,
    persist_dir: Optional[Path] = None,
    where: Optional[dict] = None,
) -> list[QueryHit]:
    """Top-K dense retrieval against the persistent collection."""
    try:
        import chromadb
    except ImportError as e:
        raise ImportError("chromadb not installed") from e

    from src.llm.gemini_client import get_default_client

    persist = Path(persist_dir or _persist_dir_for(tender_id))
    if not persist.exists():
        return []

    chroma = chromadb.PersistentClient(path=str(persist))
    try:
        collection = chroma.get_collection(name=_collection_name(tender_id))
    except Exception:
        return []

    client = get_default_client()
    q_emb_list, _ = client.embed([query_text])
    q_emb = q_emb_list[0]

    try:
        result = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
        )
    except Exception as e:
        logger.warning("Chroma query failed: %s", e)
        return []

    hits: list[QueryHit] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), 1):
        # ChromaDB returns L2 distance by default; convert to a similarity-like
        # score in (0, 1] for caller convenience.
        score = 1.0 / (1.0 + float(dist))
        hits.append(QueryHit(
            clause_id=cid,
            score=score,
            text=doc,
            metadata=meta or {},
            rank=rank,
        ))
    return hits


# ─── Inspection ─────────────────────────────────────────────────────────────
def collection_summary(tender_id: str, persist_dir: Optional[Path] = None) -> dict:
    """Return a summary dict for the collection (count + first few IDs)."""
    try:
        import chromadb
    except ImportError:
        return {"error": "chromadb not installed"}

    persist = Path(persist_dir or _persist_dir_for(tender_id))
    if not persist.exists():
        return {"exists": False}
    chroma = chromadb.PersistentClient(path=str(persist))
    try:
        coll = chroma.get_collection(name=_collection_name(tender_id))
    except Exception:
        return {"exists": False}

    return {
        "exists": True,
        "name": _collection_name(tender_id),
        "count": coll.count(),
        "persist_dir": str(persist),
    }
