"""Multi-backend LLM client with automatic fallback on quota errors.

Order:
  1. Gemini with the primary API key (env var GEMINI_API_KEY)
  2. Groq with the Groq API key (env var GROQ_API_KEY) — LLM only
  3. Gemini with the fallback API key (env var GEMINI_API_KEY_OLD)

When a backend returns a 429 / RESOURCE_EXHAUSTED / quota error, it is
marked as "cooled-down" for 60 seconds and the next backend in the chain
is tried. Embeddings only support Gemini backends — Groq has no embed
endpoint, so embed() falls back from key 1 to key 2 directly.

The wrapper presents the same surface as `GeminiClient`:

    client = get_fallback_client()
    parsed, stats = client.generate(prompt, response_schema=Schema)
    vecs, stats = client.embed(["hello", "world"])

Stats include `backend_used` (e.g. "gemini-key1" / "groq" / "gemini-key2")
so manifests record which provider answered each call.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from src.llm.gemini_client import GeminiClient, LLMCallStats, USD_TO_INR

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


# ─── Cooldown registry ─────────────────────────────────────────────────────
_BACKEND_COOLDOWN: dict[str, float] = {}
_COOLDOWN_SECONDS = 60.0


def _is_cooled_down(name: str) -> bool:
    until = _BACKEND_COOLDOWN.get(name, 0.0)
    return time.monotonic() < until


def _trip_cooldown(name: str, seconds: float = _COOLDOWN_SECONDS) -> None:
    _BACKEND_COOLDOWN[name] = time.monotonic() + seconds
    logger.warning("Backend %s tripped on quota — cooling down %.0fs", name, seconds)


def _is_quota_error(exc: Exception) -> bool:
    """Detect quota / rate-limit errors regardless of backend."""
    msg = str(exc).lower()
    return any(s in msg for s in (
        "429", "rate limit", "resource_exhausted", "quota",
        "too many requests", "exceeded", "free_tier_requests",
    ))


# ─── Multi-backend client ──────────────────────────────────────────────────
class FallbackLLMClient:
    """Wraps multiple LLM backends with quota-aware fallback."""

    def __init__(self) -> None:
        # Gemini primary
        self._gemini_primary: Optional[GeminiClient] = None
        primary_key = os.environ.get("GEMINI_API_KEY", "")
        if primary_key and not primary_key.startswith("YOUR_"):
            try:
                self._gemini_primary = GeminiClient(api_key=primary_key)
                logger.info("Initialised Gemini primary backend.")
            except Exception as e:
                logger.warning("Gemini primary init failed: %s", e)

        # Gemini fallback (older paid key)
        self._gemini_fallback: Optional[GeminiClient] = None
        fallback_key = os.environ.get("GEMINI_API_KEY_OLD", "")
        if fallback_key and not fallback_key.startswith("YOUR_") and fallback_key != primary_key:
            try:
                self._gemini_fallback = GeminiClient(api_key=fallback_key)
                logger.info("Initialised Gemini fallback backend.")
            except Exception as e:
                logger.warning("Gemini fallback init failed: %s", e)

        # Groq middle tier (LLM only)
        self._groq_key = os.environ.get("GROQ_API_KEY", "")
        self._groq_client = None
        if self._groq_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self._groq_key)
                logger.info("Initialised Groq backend.")
            except ImportError:
                logger.warning("Groq SDK not installed; install `groq` to enable fallback.")
            except Exception as e:
                logger.warning("Groq init failed: %s", e)

        # Default model exports — match GeminiClient API surface
        self.default_flash = (
            self._gemini_primary.default_flash if self._gemini_primary
            else self._gemini_fallback.default_flash if self._gemini_fallback
            else "gemini-2.5-flash"
        )
        self.default_pro = (
            self._gemini_primary.default_pro if self._gemini_primary
            else self._gemini_fallback.default_pro if self._gemini_fallback
            else "gemini-2.5-pro"
        )
        self.default_embedding = (
            self._gemini_primary.default_embedding if self._gemini_primary
            else self._gemini_fallback.default_embedding if self._gemini_fallback
            else "gemini-embedding-001"
        )

        # Expose underlying _client for any caller that pokes at it (the
        # parser's Vision-OCR fallback does this). Defaults to primary;
        # callers should prefer .generate / .embed.
        if self._gemini_primary:
            self._client = self._gemini_primary._client
        elif self._gemini_fallback:
            self._client = self._gemini_fallback._client
        else:
            self._client = None

        self.use_vertex = False

    # ──────────────────────────────────────────────────────────────────────
    # generate(): try Gemini-1 → Groq → Gemini-2
    # ──────────────────────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        response_schema: Optional[type[SchemaT]] = None,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        force_refresh: bool = False,
    ) -> tuple[Any, LLMCallStats]:
        last_exc: Optional[Exception] = None

        # Try Gemini primary
        if self._gemini_primary and not _is_cooled_down("gemini-primary"):
            try:
                parsed, stats = self._gemini_primary.generate(
                    prompt,
                    model=model,
                    response_schema=response_schema,
                    temperature=temperature,
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                    force_refresh=force_refresh,
                )
                return parsed, stats
            except Exception as e:
                last_exc = e
                if _is_quota_error(e):
                    _trip_cooldown("gemini-primary")
                else:
                    logger.warning("Gemini primary failed (non-quota): %s", e)
                    raise

        # Try Groq
        if self._groq_client and not _is_cooled_down("groq"):
            try:
                parsed, stats = self._call_groq(
                    prompt, response_schema=response_schema,
                    temperature=temperature, system_instruction=system_instruction,
                )
                return parsed, stats
            except Exception as e:
                last_exc = e
                if _is_quota_error(e):
                    _trip_cooldown("groq")
                else:
                    logger.warning("Groq failed (non-quota): %s", e)

        # Try Gemini fallback
        if self._gemini_fallback and not _is_cooled_down("gemini-fallback"):
            try:
                parsed, stats = self._gemini_fallback.generate(
                    prompt,
                    model=model,
                    response_schema=response_schema,
                    temperature=temperature,
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                    force_refresh=force_refresh,
                )
                return parsed, stats
            except Exception as e:
                last_exc = e
                if _is_quota_error(e):
                    _trip_cooldown("gemini-fallback")
                raise

        # All backends down
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            "All LLM backends are cooled-down or unconfigured. "
            "Check GEMINI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY_OLD."
        )

    def _call_groq(
        self,
        prompt: str,
        *,
        response_schema: Optional[type[SchemaT]],
        temperature: float,
        system_instruction: Optional[str],
    ) -> tuple[Any, LLMCallStats]:
        """Call Groq with prompt-engineered JSON for structured output."""
        # Default to Llama 3.3 70B versatile — best free quality on Groq
        model = "llama-3.3-70b-versatile"

        messages = []
        sys_prompt = system_instruction or ""
        if response_schema is not None:
            schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
            sys_prompt += (
                f"\n\nYou MUST respond with valid JSON matching this schema. "
                f"Output ONLY the JSON object, no prose, no markdown fences.\n\n"
                f"Schema:\n{schema_json}"
            )
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        resp = self._groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"} if response_schema else None,
        )
        latency = time.monotonic() - t0

        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Groq Llama 3.3 70B free-tier: ₹0 to user. Estimate cost as 0.
        cost_usd = 0.0
        stats = LLMCallStats(
            model=f"groq:{model}",
            cache_hit=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            cost_inr=cost_usd * USD_TO_INR,
            latency_seconds=latency,
            response_text_length=len(text),
        )

        if response_schema is not None:
            try:
                parsed_dict = json.loads(text)
                parsed = response_schema.model_validate(parsed_dict)
                return parsed, stats
            except Exception as e:
                raise ValueError(
                    f"Groq output could not be parsed as {response_schema.__name__}: "
                    f"{e}\nRaw response: {text[:500]}"
                )
        return text, stats

    # ──────────────────────────────────────────────────────────────────────
    # embed(): Gemini-only fallback chain (Groq has no embed API)
    # ──────────────────────────────────────────────────────────────────────
    def embed(
        self,
        texts: list[str],
        *,
        model: Optional[str] = None,
        force_refresh: bool = False,
    ) -> tuple[list[list[float]], LLMCallStats]:
        last_exc: Optional[Exception] = None

        for name, client in [
            ("gemini-primary", self._gemini_primary),
            ("gemini-fallback", self._gemini_fallback),
        ]:
            if client is None or _is_cooled_down(name):
                continue
            try:
                return client.embed(texts, model=model, force_refresh=force_refresh)
            except Exception as e:
                last_exc = e
                if _is_quota_error(e):
                    _trip_cooldown(name)
                else:
                    raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            "All embedding backends cooled-down. "
            "Check GEMINI_API_KEY and GEMINI_API_KEY_OLD."
        )

    # ── Pass-through accessors used by manifests / parser hooks ──────────
    @property
    def cache_disabled(self) -> bool:
        if self._gemini_primary:
            return self._gemini_primary.cache_disabled
        if self._gemini_fallback:
            return self._gemini_fallback.cache_disabled
        return False

    @property
    def totals(self) -> LLMCallStats:
        # Sum primary + fallback totals (Groq totals are not aggregated since
        # cost is zero anyway).
        agg = LLMCallStats(model="<aggregate>", cache_hit=False)
        for c in (self._gemini_primary, self._gemini_fallback):
            if c is None:
                continue
            t = c.totals
            agg.prompt_tokens += t.prompt_tokens
            agg.completion_tokens += t.completion_tokens
            agg.total_tokens += t.total_tokens
            agg.cost_usd += t.cost_usd
            agg.cost_inr += t.cost_inr
            agg.latency_seconds += t.latency_seconds
        return agg

    def reset_totals(self) -> None:
        if self._gemini_primary:
            self._gemini_primary.reset_totals()
        if self._gemini_fallback:
            self._gemini_fallback.reset_totals()


# ─── Module-level helper ───────────────────────────────────────────────────
_default_fallback: Optional[FallbackLLMClient] = None


def get_fallback_client() -> FallbackLLMClient:
    """Return a process-wide shared FallbackLLMClient (lazy-initialised)."""
    global _default_fallback
    if _default_fallback is None:
        _default_fallback = FallbackLLMClient()
    return _default_fallback


__all__ = ["FallbackLLMClient", "get_fallback_client"]
