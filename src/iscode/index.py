"""Build and query the IS-code corpus vector index.

ChromaDB collection `iscode_corpus` lives at .cache/chroma/iscode_corpus/
(separate from per-tender collections). One-time embedding pass over every
PDF in resources/iscode/. Idempotent — re-runs embed only new chunks.

Also provides a deterministic citation lookup: given a (code_id, version,
section) tuple, return the matching ISCodeChunk if it exists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.iscode.parser import parse_iscode_pdf
from src.schemas.rewrite import ISCodeChunk

logger = logging.getLogger(__name__)


COLLECTION_NAME = "iscode_corpus"


# ─── Path helpers ──────────────────────────────────────────────────────────
def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _persist_dir() -> Path:
    return _project_root() / ".cache" / "chroma" / COLLECTION_NAME


def _resources_dir() -> Path:
    return _project_root() / "resources" / "iscode"


def _registry_path() -> Path:
    """Plain-JSON registry for fast deterministic citation lookup."""
    return _project_root() / ".cache" / "iscode_registry.json"


# ─── Build ─────────────────────────────────────────────────────────────────
@dataclass
class BuildResult:
    pdfs_seen: int
    chunks_total: int
    newly_embedded: int
    skipped_already_indexed: int
    cost_usd: float
    cost_inr: float


def build_iscode_index(
    *,
    resources_root: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
    embed_batch_size: int = 32,
) -> BuildResult:
    """Build (or update) the IS-code corpus index.

    Iterates every *.pdf in resources/iscode/, parses + chunks, and embeds
    chunks not already in the collection. Also writes a flat registry
    (.cache/iscode_registry.json) keyed by (code_id, section) for fast
    citation lookup.
    """
    try:
        import chromadb
    except ImportError as e:
        raise ImportError("chromadb not installed") from e

    from src.llm.gemini_client import get_default_client

    resources_root = Path(resources_root or _resources_dir())
    persist_dir = Path(persist_dir or _persist_dir())
    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
    existing_ids: set[str] = set(collection.get()["ids"] or [])

    pdfs = sorted(resources_root.glob("*.pdf"))
    logger.info("Found %d IS-code / CPWD PDFs in %s", len(pdfs), resources_root)

    all_chunks: list[ISCodeChunk] = []
    for pdf in pdfs:
        logger.info("Parsing %s...", pdf.name)
        try:
            chunks = parse_iscode_pdf(pdf)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", pdf.name, e)
            continue
        logger.info("  → %d chunks", len(chunks))
        all_chunks.extend(chunks)

    # Filter to chunks not already in the collection
    to_embed = [c for c in all_chunks if c.chunk_id not in existing_ids]
    skipped = len(all_chunks) - len(to_embed)

    cost_usd = 0.0
    cost_inr = 0.0
    newly = 0

    if to_embed:
        client = get_default_client()
        for i in range(0, len(to_embed), embed_batch_size):
            batch = to_embed[i : i + embed_batch_size]
            texts = [c.text for c in batch]
            embeds, stats = client.embed(texts)
            cost_usd += stats.cost_usd
            cost_inr += stats.cost_inr

            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeds,
                documents=texts,
                metadatas=[_metadata_for(c) for c in batch],
            )
            newly += len(batch)
            logger.info("Embedded batch %d-%d / %d (cumulative cost INR %.4f)",
                        i, i + len(batch), len(to_embed), cost_inr)

    # Write the deterministic registry
    _write_registry(all_chunks, _registry_path())

    return BuildResult(
        pdfs_seen=len(pdfs),
        chunks_total=len(all_chunks),
        newly_embedded=newly,
        skipped_already_indexed=skipped,
        cost_usd=cost_usd,
        cost_inr=cost_inr,
    )


def _metadata_for(c: ISCodeChunk) -> dict:
    return {
        "code_id": c.code_id,
        "code_title": c.code_title,
        "version_year": c.version_year if c.version_year is not None else 0,
        "section": c.section,
        "page": c.page,
        "char_count": c.char_count,
        "source_file": c.source_file,
    }


def _write_registry(chunks: list[ISCodeChunk], out: Path) -> None:
    """A small JSON keyed by 'code_id::section' for fast lookup by the
    citation verifier."""
    out.parent.mkdir(parents=True, exist_ok=True)
    registry: dict[str, dict] = {}
    for c in chunks:
        # Multiple chunks may share a section if it was hard-split. Keep the
        # first occurrence (earliest page).
        key = f"{c.code_id}::{c.section}"
        if key in registry:
            continue
        registry[key] = {
            "chunk_id": c.chunk_id,
            "code_id": c.code_id,
            "code_title": c.code_title,
            "version_year": c.version_year,
            "section": c.section,
            "page": c.page,
            "source_file": c.source_file,
        }
    with out.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def load_registry() -> dict[str, dict]:
    p = _registry_path()
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ─── Query ─────────────────────────────────────────────────────────────────
def query_iscode_index(
    query_text: str,
    *,
    top_k: int = 6,
    persist_dir: Optional[Path] = None,
) -> list[ISCodeChunk]:
    """Top-K dense retrieval against the IS-code corpus."""
    try:
        import chromadb
    except ImportError as e:
        raise ImportError("chromadb not installed") from e

    from src.llm.gemini_client import get_default_client

    persist_dir = Path(persist_dir or _persist_dir())
    if not persist_dir.exists():
        return []

    chroma = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = chroma.get_collection(name=COLLECTION_NAME)
    except Exception:
        return []

    client = get_default_client()
    try:
        q_emb_list, _ = client.embed([query_text])
        q_emb = q_emb_list[0]
    except Exception as e:
        logger.warning(
            "IS-code retrieval skipped: embed call failed (%s). "
            "Stage 4 will continue without IS-code context, but G1 will reject "
            "any rewrite that needs grounding.", e,
        )
        return []

    try:
        result = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
        )
    except Exception as e:
        logger.warning("Chroma query failed: %s", e)
        return []

    out: list[ISCodeChunk] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    for cid, doc, meta in zip(ids, docs, metas):
        m = meta or {}
        out.append(ISCodeChunk(
            chunk_id=cid,
            code_id=m.get("code_id", ""),
            code_title=m.get("code_title", ""),
            version_year=m.get("version_year") if m.get("version_year") else None,
            section=m.get("section", ""),
            page=int(m.get("page", 0)),
            text=doc,
            char_count=int(m.get("char_count", len(doc))),
            source_file=m.get("source_file", ""),
        ))
    return out


# ─── Summary ───────────────────────────────────────────────────────────────
def index_summary() -> dict:
    """Return summary of the IS-code index state."""
    try:
        import chromadb
    except ImportError:
        return {"error": "chromadb not installed"}

    persist = _persist_dir()
    if not persist.exists():
        return {"exists": False}
    chroma = chromadb.PersistentClient(path=str(persist))
    try:
        coll = chroma.get_collection(name=COLLECTION_NAME)
    except Exception:
        return {"exists": False}
    return {
        "exists": True,
        "name": COLLECTION_NAME,
        "count": coll.count(),
        "persist_dir": str(persist),
        "registry_size": len(load_registry()),
    }
