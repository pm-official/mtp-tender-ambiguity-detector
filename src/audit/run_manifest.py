"""Run manifest — per-pipeline-invocation reproducibility record.

Every time a stage of the pipeline runs, we record:
  * Stage name + version
  * Models used (with full version strings)
  * Parameters (thresholds, top-K, batch sizes, ...)
  * Inputs (file paths + SHA256 of source artefacts)
  * Outputs (file paths + line counts)
  * Costs in USD and INR
  * Timing
  * Environment snapshot (Python version, package versions of LLM-touching libs)
  * Random seeds where applicable

The manifest is appended to `corpus/<tender_id>/runs/<timestamp>.json` and
indexed in `corpus/<tender_id>/runs/_index.jsonl` so a paper reviewer can
trace any reported number back to the exact configuration that produced it.

This addresses the methodology document's reproducibility requirements
(Appendix C.4 reproducibility checklist) and is critical for conference-
paper review where reviewers ask "rerun this with the parameters in your
methodology table — do you get the same numbers?".
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import platform
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Bumped when the manifest schema changes incompatibly.
MANIFEST_SCHEMA_VERSION = "1.0"

# Pipeline stage version markers. Bump when stage logic changes meaningfully.
STAGE_VERSIONS = {
    "stage_0_parse": "1.1",          # bumped after the multiline-anchor fix in chunker
    "stage_0_extract": "1.0",
    "stage_0_silver_labels": "0.0",  # not yet built (Session 5)
    "stage_1_bct": "1.0",
    "stage_2_critic": "1.1",         # bumped when default model went Flash → Pro
    "stage_3_critique_severity": "0.0",  # not yet built (Session 8)
    "stage_4_rewrite": "0.0",            # not yet built (Session 9)
}


# ─── Data class ─────────────────────────────────────────────────────────────
@dataclass
class RunManifest:
    """One pipeline-stage invocation's full provenance record."""

    run_id: str                              # uuid4
    tender_id: str
    stage: str                               # "stage_1_bct", "stage_2_critic", etc.
    stage_version: str
    schema_version: str = MANIFEST_SCHEMA_VERSION

    started_at: str = ""                     # ISO 8601 UTC
    finished_at: str = ""
    seconds_elapsed: float = 0.0
    success: bool = False
    error: Optional[str] = None

    models: dict[str, str] = field(default_factory=dict)         # role → model_name
    parameters: dict[str, Any] = field(default_factory=dict)
    inputs: list[dict[str, Any]] = field(default_factory=list)   # [{path, sha256, lines}]
    outputs: list[dict[str, Any]] = field(default_factory=list)

    cost_usd: float = 0.0
    cost_inr: float = 0.0
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    # Stage-specific summary numbers (so the manifest is a self-contained
    # record without needing to re-open output files).
    summary: dict[str, Any] = field(default_factory=dict)

    environment: dict[str, str] = field(default_factory=dict)
    random_seed: Optional[int] = None
    cli_args: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Lifecycle helpers ──────────────────────────────────────────────────────
def start_run(
    *,
    tender_id: str,
    stage: str,
    parameters: Optional[dict[str, Any]] = None,
    models: Optional[dict[str, str]] = None,
    random_seed: Optional[int] = None,
    cli_args: Optional[list[str]] = None,
) -> RunManifest:
    """Open a new run manifest. Populate inputs/outputs/summary as the run progresses."""
    return RunManifest(
        run_id=str(uuid.uuid4()),
        tender_id=tender_id,
        stage=stage,
        stage_version=STAGE_VERSIONS.get(stage, "0.0"),
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        models=dict(models or {}),
        parameters=dict(parameters or {}),
        environment=_capture_environment(),
        random_seed=random_seed,
        cli_args=list(cli_args or sys.argv[1:]),
    )


def finish_run(
    manifest: RunManifest,
    *,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """Close the run, capture finish time, persist to disk."""
    manifest.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        t0 = datetime.fromisoformat(manifest.started_at)
        t1 = datetime.fromisoformat(manifest.finished_at)
        manifest.seconds_elapsed = (t1 - t0).total_seconds()
    except Exception:
        pass
    manifest.success = success
    manifest.error = error


def record_input(manifest: RunManifest, path: Path, *, kind: str = "file") -> None:
    """Record a file input with its SHA256 and (if jsonl) line count."""
    p = Path(path)
    entry: dict[str, Any] = {"path": str(p), "kind": kind}
    if p.exists() and p.is_file():
        entry["size_bytes"] = p.stat().st_size
        entry["sha256"] = _sha256_file(p)
        if p.suffix.lower() == ".jsonl":
            entry["lines"] = sum(1 for _ in p.open("r", encoding="utf-8"))
    else:
        entry["missing"] = True
    manifest.inputs.append(entry)


def record_output(manifest: RunManifest, path: Path, *, kind: str = "file") -> None:
    """Record a file output (after it is written). Same shape as record_input."""
    p = Path(path)
    entry: dict[str, Any] = {"path": str(p), "kind": kind}
    if p.exists() and p.is_file():
        entry["size_bytes"] = p.stat().st_size
        entry["sha256"] = _sha256_file(p)
        if p.suffix.lower() == ".jsonl":
            entry["lines"] = sum(1 for _ in p.open("r", encoding="utf-8"))
    else:
        entry["missing"] = True
    manifest.outputs.append(entry)


def record_llm_stats(manifest: RunManifest, stats) -> None:
    """Accumulate LLM telemetry into the manifest. `stats` is an LLMCallStats."""
    manifest.api_calls += 1
    if getattr(stats, "cache_hit", False):
        manifest.cache_hits += 1
    else:
        manifest.cache_misses += 1
    manifest.cost_usd += getattr(stats, "cost_usd", 0.0) or 0.0
    manifest.cost_inr += getattr(stats, "cost_inr", 0.0) or 0.0
    manifest.tokens_in += getattr(stats, "prompt_tokens", 0) or 0
    manifest.tokens_out += getattr(stats, "completion_tokens", 0) or 0


def write_manifest(tender_dir: Path, manifest: RunManifest) -> Path:
    """Persist manifest to corpus/<tender_id>/runs/<timestamp>_<stage>.json
    and append a one-line summary to runs/_index.jsonl."""
    runs_dir = Path(tender_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = manifest.started_at.replace(":", "-").replace("+00-00", "Z")
    out = runs_dir / f"{ts}__{manifest.stage}__{manifest.run_id[:8]}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest.as_dict(), f, indent=2, ensure_ascii=False, default=str)
    # Index entry — one line per run for fast `cat _index.jsonl | jq` queries
    index_entry = {
        "run_id": manifest.run_id,
        "stage": manifest.stage,
        "stage_version": manifest.stage_version,
        "started_at": manifest.started_at,
        "finished_at": manifest.finished_at,
        "seconds_elapsed": round(manifest.seconds_elapsed, 1),
        "success": manifest.success,
        "cost_usd": round(manifest.cost_usd, 6),
        "cost_inr": round(manifest.cost_inr, 4),
        "api_calls": manifest.api_calls,
        "cache_hit_rate": round(manifest.cache_hits / max(manifest.api_calls, 1), 3),
        "manifest_path": str(out.relative_to(tender_dir)),
        "summary": manifest.summary,
    }
    with (runs_dir / "_index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
    return out


def list_runs(tender_dir: Path) -> list[dict]:
    """Return the contents of runs/_index.jsonl, one dict per past run."""
    p = Path(tender_dir) / "runs" / "_index.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


# ─── Environment capture ───────────────────────────────────────────────────
def _capture_environment() -> dict[str, str]:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": _safe(socket.gethostname),
        "user": _safe(getpass.getuser),
    }
    # Capture versions of the LLM-touching libraries
    for pkg in ("google-genai", "google-generativeai", "pydantic", "chromadb",
                "pymupdf", "rapidfuzz", "scikit-learn", "tenacity"):
        env[f"pkg_{pkg}"] = _package_version(pkg)
    # Git revision if running in a repo
    git = _safe(_git_short_sha)
    if git:
        env["git_sha"] = git
    return env


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def _git_short_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def _sha256_file(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()
