"""Live smoke test against the Gemini API.

This test makes a SMALL real API call to verify (a) the API key works, (b) the
client can do structured output, (c) the response parses into a Pydantic model.

It is automatically skipped if GEMINI_API_KEY is unset or set to the placeholder.

Run:
    pytest tests/test_gemini_smoke.py -v -s
"""

from __future__ import annotations

import os
import re

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Make sure .env is loaded BEFORE we read the key. Use override=True so we
# trump any dummy value that the offline test module may have left in env
# during pytest collection.
load_dotenv(override=True)

from src.llm.gemini_client import GeminiClient, get_default_client  # noqa: E402


# ─── Skip if no real API key ────────────────────────────────────────────────
_KEY = os.environ.get("GEMINI_API_KEY", "")
_SKIP_REASON = (
    "GEMINI_API_KEY not set or still set to placeholder. "
    "Copy .env.example to .env and add your real key from "
    "https://aistudio.google.com/app/apikey"
)
_HAVE_KEY = bool(_KEY) and not _KEY.startswith("YOUR_") and _KEY != "dummy-test-key-NOT-REAL"

pytestmark = pytest.mark.skipif(not _HAVE_KEY, reason=_SKIP_REASON)


# ─── Schema for the smoke test ──────────────────────────────────────────────
class CityFact(BaseModel):
    city: str = Field(..., description="A real city name")
    country: str = Field(..., description="The country it is in")
    famous_for: str = Field(..., description="One short fact (≤ 12 words)")


# ─── Tests ──────────────────────────────────────────────────────────────────
def test_structured_output_round_trip():
    """Single Gemini Flash call returning a parsed Pydantic object."""
    client = get_default_client()
    try:
        parsed, stats = client.generate(
            prompt=(
                "Return a JSON object describing the city of Mumbai with three "
                "fields: city, country, famous_for. Keep famous_for under 12 words."
            ),
            response_schema=CityFact,
            model="gemini-2.5-flash",
            temperature=0.0,
        )
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ("429", "403", "RESOURCE_EXHAUSTED", "PERMISSION_DENIED",
                                   "RemoteProtocolError", "ConnectError",
                                   "ReadTimeout", "ConnectTimeout")):
            pytest.skip(f"Transient API error / quota / project block: {e}")
        raise
    print(f"\n  -> response: {parsed!r}")
    print(
        f"  -- tokens in/out/total: "
        f"{stats.prompt_tokens}/{stats.completion_tokens}/{stats.total_tokens}"
    )
    print(f"  -- cost USD: {stats.cost_usd:.6f}  (~ INR {stats.cost_inr:.4f})")
    print(f"  -- cache hit: {stats.cache_hit}")
    print(f"  -- latency: {stats.latency_seconds:.2f}s")

    # Assertions
    assert isinstance(parsed, CityFact), "Should be parsed into CityFact"
    assert re.search(r"mumbai", parsed.city, re.IGNORECASE)
    assert parsed.country.strip() != ""
    assert parsed.famous_for.strip() != ""
    # Cache state is exercised by test_cache_hit_on_second_call below.
    # Here we only assert the response shape and cost is non-negative.
    assert stats.cost_usd >= 0.0


def test_cache_hit_on_second_call():
    """Repeated identical call must hit the cache and skip the API."""
    client = GeminiClient()  # fresh client, but shared cache dir
    prompt = (
        "Return a JSON describing the city of Pune with city, country, famous_for."
    )

    # First call may hit-or-miss (depends on prior runs). Force fresh.
    try:
        parsed1, stats1 = client.generate(
            prompt=prompt,
            response_schema=CityFact,
            model="gemini-2.5-flash",
            temperature=0.0,
            force_refresh=True,
        )
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ("429", "403", "RESOURCE_EXHAUSTED", "PERMISSION_DENIED",
                                   "RemoteProtocolError", "ConnectError",
                                   "ReadTimeout", "ConnectTimeout")):
            pytest.skip(f"Transient API error / quota / project block: {e}")
        raise
    assert stats1.cache_hit is False

    # Second call must come from cache
    parsed2, stats2 = client.generate(
        prompt=prompt,
        response_schema=CityFact,
        model="gemini-2.5-flash",
        temperature=0.0,
    )
    assert stats2.cache_hit is True, "Second identical call must hit the cache"
    assert stats2.cost_usd == 0.0
    # Same content
    assert parsed1.model_dump() == parsed2.model_dump()


def test_freeform_text_response():
    """Calls without a schema should return a plain string."""
    client = get_default_client()
    try:
        text, stats = client.generate(
            prompt="Reply with the single word: hello",
            model="gemini-2.5-flash",
            temperature=0.0,
        )
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ("429", "403", "RESOURCE_EXHAUSTED", "PERMISSION_DENIED",
                                   "RemoteProtocolError", "ConnectError",
                                   "ReadTimeout", "ConnectTimeout")):
            pytest.skip(f"Transient API error / quota / project block: {e}")
        raise
    assert isinstance(text, str)
    assert "hello" in text.lower()
    print(f"\n  -> text: {text!r}")
    print(f"  -- cache hit: {stats.cache_hit}")
