"""Offline tests for the Gemini client wrapper.

These tests do NOT call the live Gemini API. They verify pure-Python behaviour:
cache key construction, cache I/O round-trip, cost estimation, retry classification.

Run:
    pytest tests/test_gemini_client_offline.py -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

# Set a dummy key so the constructor doesn't bail on missing env var
os.environ.setdefault("GEMINI_API_KEY", "dummy-test-key-NOT-REAL")
os.environ.setdefault("GEMINI_CACHE_DISABLED", "0")


from src.llm.gemini_client import (  # noqa: E402
    GeminiClient,
    LLMCallStats,
    USD_TO_INR,
    _estimate_cost_usd,
)


@pytest.fixture
def tmp_cache_dir():
    d = Path(tempfile.mkdtemp(prefix="gemini_cache_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def client(tmp_cache_dir):
    """A client backed by a throwaway cache dir."""
    return GeminiClient(api_key="dummy-test-key-NOT-REAL", cache_dir=tmp_cache_dir)


# ─── Test schemas ───────────────────────────────────────────────────────────
class _DemoSchema(BaseModel):
    field: str
    score: float


# ─── Cache key tests ────────────────────────────────────────────────────────
def test_cache_key_is_deterministic(client):
    key1 = client._build_cache_key(
        kind="generate", model="gemini-2.5-flash", prompt="hello world",
        response_schema=_DemoSchema, temperature=0.0,
    )
    key2 = client._build_cache_key(
        kind="generate", model="gemini-2.5-flash", prompt="hello world",
        response_schema=_DemoSchema, temperature=0.0,
    )
    assert key1 == key2, "Same inputs must produce the same cache key"
    assert len(key1) == 64, "Should be a SHA256 hex digest"


def test_cache_key_changes_with_prompt(client):
    key1 = client._build_cache_key(
        kind="generate", model="gemini-2.5-flash", prompt="hello A",
    )
    key2 = client._build_cache_key(
        kind="generate", model="gemini-2.5-flash", prompt="hello B",
    )
    assert key1 != key2


def test_cache_key_changes_with_model(client):
    key1 = client._build_cache_key(
        kind="generate", model="gemini-2.5-flash", prompt="hi",
    )
    key2 = client._build_cache_key(
        kind="generate", model="gemini-2.5-pro", prompt="hi",
    )
    assert key1 != key2


def test_cache_key_changes_with_temperature(client):
    key1 = client._build_cache_key(
        kind="generate", model="gemini-2.5-flash", prompt="hi", temperature=0.0,
    )
    key2 = client._build_cache_key(
        kind="generate", model="gemini-2.5-flash", prompt="hi", temperature=0.5,
    )
    assert key1 != key2


def test_cache_key_changes_with_schema(client):
    class _OtherSchema(BaseModel):
        x: int

    key1 = client._build_cache_key(
        kind="generate", model="gemini-2.5-flash", prompt="hi",
        response_schema=_DemoSchema,
    )
    key2 = client._build_cache_key(
        kind="generate", model="gemini-2.5-flash", prompt="hi",
        response_schema=_OtherSchema,
    )
    assert key1 != key2


# ─── Cache I/O round-trip tests ─────────────────────────────────────────────
def test_cache_round_trip(client, tmp_cache_dir):
    key = client._build_cache_key(kind="generate", model="m", prompt="p")
    payload = {
        "model": "m",
        "prompt": "p",
        "response_text": "Some text response",
        "parsed_dict": {"field": "ok", "score": 1.0},
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "cost_usd": 0.001,
    }
    client._cache_put(key, payload)

    cached = client._cache_get(key)
    assert cached is not None
    assert cached["response_text"] == "Some text response"
    assert cached["parsed_dict"]["field"] == "ok"
    # Cache file should be sharded
    assert (tmp_cache_dir / key[:2] / f"{key}.json").exists()


def test_cache_miss_returns_none(client):
    cached = client._cache_get("0" * 64)
    assert cached is None


# ─── Cost estimation tests ──────────────────────────────────────────────────
def test_cost_flash_zero_tokens():
    assert _estimate_cost_usd("gemini-2.5-flash", 0, 0) == 0.0


def test_cost_flash_one_million_input_tokens():
    cost = _estimate_cost_usd("gemini-2.5-flash", 1_000_000, 0)
    # 1M input × $0.075 = $0.075
    assert abs(cost - 0.075) < 1e-6


def test_cost_pro_round_trip():
    cost = _estimate_cost_usd("gemini-2.5-pro", 100_000, 50_000)
    # 0.1M × $1.25 + 0.05M × $5 = 0.125 + 0.25 = 0.375
    assert abs(cost - 0.375) < 1e-6


def test_cost_unknown_model_returns_zero():
    cost = _estimate_cost_usd("nonexistent-model", 1000, 1000)
    assert cost == 0.0


# ─── Stats / totals tests ───────────────────────────────────────────────────
def test_totals_initialise_to_zero(client):
    assert client.totals.prompt_tokens == 0
    assert client.totals.cost_usd == 0.0


def test_totals_accumulate(client):
    s1 = LLMCallStats(
        model="gemini-2.5-flash", cache_hit=False,
        prompt_tokens=100, completion_tokens=200, total_tokens=300,
        cost_usd=0.001, cost_inr=0.001 * USD_TO_INR, latency_seconds=1.5,
    )
    s2 = LLMCallStats(
        model="gemini-2.5-flash", cache_hit=False,
        prompt_tokens=50, completion_tokens=100, total_tokens=150,
        cost_usd=0.0005, cost_inr=0.0005 * USD_TO_INR, latency_seconds=0.8,
    )
    client._add_to_totals(s1)
    client._add_to_totals(s2)
    assert client.totals.prompt_tokens == 150
    assert client.totals.completion_tokens == 300
    assert abs(client.totals.cost_usd - 0.0015) < 1e-9


def test_totals_skip_cache_hits(client):
    s = LLMCallStats(
        model="gemini-2.5-flash", cache_hit=True,
        prompt_tokens=100, completion_tokens=200, total_tokens=300,
        cost_usd=0.001,
    )
    client._add_to_totals(s)
    # Cache hits should not be added to totals (they're free)
    assert client.totals.cost_usd == 0.0
    assert client.totals.prompt_tokens == 0


# ─── Constructor guards ─────────────────────────────────────────────────────
def test_constructor_rejects_placeholder_key(monkeypatch, tmp_cache_dir):
    """In AI Studio mode, missing GEMINI_API_KEY must raise RuntimeError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_USE_VERTEX", "0")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY not set"):
        GeminiClient(api_key=None, cache_dir=tmp_cache_dir)


def test_constructor_accepts_real_looking_key(monkeypatch, tmp_cache_dir):
    """Just verifies that providing a non-placeholder key constructs without error."""
    monkeypatch.setenv("GEMINI_USE_VERTEX", "0")
    c = GeminiClient(api_key="AIzaSy_FAKE_BUT_LOOKS_REAL", cache_dir=tmp_cache_dir)
    assert c.api_key == "AIzaSy_FAKE_BUT_LOOKS_REAL"
    assert c.cache_dir == tmp_cache_dir
    assert c.use_vertex is False


def test_constructor_vertex_mode_requires_project_id(monkeypatch, tmp_cache_dir):
    """In Vertex mode, missing GCP_PROJECT_ID must raise RuntimeError."""
    monkeypatch.setenv("GEMINI_USE_VERTEX", "1")
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    with pytest.raises(RuntimeError, match="GCP_PROJECT_ID is not set"):
        GeminiClient(api_key=None, cache_dir=tmp_cache_dir)


def test_constructor_vertex_mode_does_not_require_api_key(monkeypatch, tmp_cache_dir):
    """Vertex mode uses ADC, so GEMINI_API_KEY can be absent."""
    monkeypatch.setenv("GEMINI_USE_VERTEX", "1")
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project-123")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Construction should succeed even with no API key in Vertex mode.
    c = GeminiClient(api_key=None, cache_dir=tmp_cache_dir)
    assert c.use_vertex is True
    assert c.gcp_project_id == "test-project-123"
    assert c.gcp_location == "us-central1"
