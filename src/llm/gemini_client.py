"""Gemini client wrapper with on-disk caching, retry, and Pydantic-typed
structured output.

Design goals (per Methodology Part 3 — design principles):
  - P1 Decisions deterministic given model output → all calls go through
       Pydantic-typed structured output where possible.
  - P2 Auditable → stats and raw responses are returned alongside parsed data.
  - P6 Cost-bounded → SHA256-keyed disk cache makes re-runs free; retry only
       on transient errors.
  - P7 Buildable in Claude Code → minimal external surface area, single
       entry-point class.

The cache is content-addressed by the hash of (model, prompt, schema, temp,
max_output_tokens, system_instruction). Two identical calls always hit the
cache; changing any input invalidates it.

Usage:
    from src.llm.gemini_client import get_default_client
    from pydantic import BaseModel

    class MySchema(BaseModel):
        field: str
        score: float

    client = get_default_client()
    parsed, stats = client.generate(
        prompt="...",
        response_schema=MySchema,
        model="gemini-2.5-flash",
    )
    print(stats.cache_hit, stats.cost_usd, parsed.field)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Load .env once on import. The .env file is gitignored; the API key never
# enters the codebase or version control.
load_dotenv()

# ─── Pricing (Gemini 2.5 published rates, May 2026; update if Google changes) ─
# Prices are USD per 1 million tokens.
PRICING_USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-lite": {"input": 0.040, "output": 0.15},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},  # legacy alias support
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},     # legacy alias support
}
# Embedding pricing is per 1 million CHARACTERS, not tokens.
EMBEDDING_PRICING_USD_PER_1M_CHARS: dict[str, float] = {
    "text-embedding-004": 0.025,
    "gemini-embedding-001": 0.025,
}

# Approximate USD → INR conversion for cost reporting. Not used for billing.
USD_TO_INR = 84.0


# ─── Stats record returned alongside every call ─────────────────────────────
@dataclass
class LLMCallStats:
    """Telemetry for a single generate / embed call."""

    model: str
    cache_hit: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_inr: float = 0.0
    latency_seconds: float = 0.0
    response_text_length: int = 0
    error: str | None = None
    # For embeddings:
    embedding_dim: int | None = None
    num_inputs: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Retryable transient errors ─────────────────────────────────────────────
class _GeminiTransientError(Exception):
    """Raised on retryable conditions (rate limit, server error, network)."""


# ─── Schema typevar ─────────────────────────────────────────────────────────
SchemaT = TypeVar("SchemaT", bound=BaseModel)


# ─── The client ─────────────────────────────────────────────────────────────
class GeminiClient:
    """Thin wrapper over google-genai with caching, retry, and cost tracking.

    Parameters
    ----------
    api_key : str | None
        If None, read from GEMINI_API_KEY environment variable.
    cache_dir : Path | None
        Directory for the on-disk response cache. Default is .cache/llm/ in the
        project root.
    cache_disabled : bool
        If True, every call is a fresh API request (cache neither read nor
        written). Useful for benchmarking and for explicitly avoiding stale
        responses.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        cache_disabled: bool | None = None,
    ) -> None:
        # ── Pick auth path: Vertex AI vs AI Studio API key ────────────────
        # Vertex AI is the credit-eligible path (uses GCP Free Trial credits
        # and any other GCP billing). AI Studio API keys bypass GCP credits
        # and bill directly against the linked payment method, so for cost-
        # sensitive runs we want Vertex. Toggle with GEMINI_USE_VERTEX=1 in
        # .env. When Vertex is on, GCP_PROJECT_ID and GCP_LOCATION are
        # required; auth is via Application Default Credentials, typically
        # a service account JSON pointed to by GOOGLE_APPLICATION_CREDENTIALS.
        self.use_vertex = os.environ.get("GEMINI_USE_VERTEX", "0") == "1"
        self.gcp_project_id = os.environ.get("GCP_PROJECT_ID", "")
        self.gcp_location = os.environ.get("GCP_LOCATION", "us-central1")

        # ── Resolve API key (only required in AI Studio mode) ─────────────
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.use_vertex:
            if not self.api_key or self.api_key.startswith("YOUR_"):
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Either:\n"
                    "  (a) Set GEMINI_USE_VERTEX=1 + GCP_PROJECT_ID in .env "
                    "to use Vertex AI (credit-eligible), or\n"
                    "  (b) Copy .env.example to .env and fill in your real "
                    "key from https://aistudio.google.com/app/apikey "
                    "(NOT credit-eligible — bills card directly)."
                )
        else:
            if not self.gcp_project_id:
                raise RuntimeError(
                    "GEMINI_USE_VERTEX=1 but GCP_PROJECT_ID is not set in .env. "
                    "Vertex mode requires the GCP project ID (e.g. "
                    "'my-gcp-project-12345')."
                )
            if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                logger.warning(
                    "Vertex mode active but GOOGLE_APPLICATION_CREDENTIALS is "
                    "not set. Calls will fall back to gcloud Application "
                    "Default Credentials. If you have not run "
                    "`gcloud auth application-default login` and have not "
                    "downloaded a service-account JSON, the API will fail."
                )

        # ── Resolve cache dir ─────────────────────────────────────────────
        env_cache = os.environ.get("GEMINI_CACHE_DIR")
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        elif env_cache:
            self.cache_dir = Path(env_cache)
        else:
            # default: <project_root>/.cache/llm/
            self.cache_dir = _project_root() / ".cache" / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ── Resolve cache enable/disable ──────────────────────────────────
        if cache_disabled is None:
            cache_disabled = os.environ.get("GEMINI_CACHE_DISABLED", "0") == "1"
        self.cache_disabled = cache_disabled

        # ── Lazily import the SDK so import errors are caught here ────────
        try:
            from google import genai

            if self.use_vertex:
                # Vertex AI mode: project + location, ADC handles auth.
                self._client = genai.Client(
                    vertexai=True,
                    project=self.gcp_project_id,
                    location=self.gcp_location,
                )
                logger.info(
                    "GeminiClient initialised in Vertex AI mode "
                    "(project=%s, location=%s).",
                    self.gcp_project_id, self.gcp_location,
                )
            else:
                self._client = genai.Client(api_key=self.api_key)
                logger.info(
                    "GeminiClient initialised in AI Studio mode "
                    "(API key path — NOT credit-eligible)."
                )
            self._genai = genai
        except ImportError as e:
            raise ImportError(
                "google-genai SDK not installed. Run: pip install google-genai"
            ) from e

        # Default models from env, can be overridden per call.
        # Note: Vertex AI accepts the same gemini-2.5-* aliases as AI Studio
        # in the modern unified SDK; only the embedding default differs.
        self.default_flash = os.environ.get("GEMINI_MODEL_FLASH", "gemini-2.5-flash")
        self.default_pro = os.environ.get("GEMINI_MODEL_PRO", "gemini-2.5-pro")
        # Keep gemini-embedding-001 across both modes — the existing
        # iscode_corpus ChromaDB collection was built with 3072-dim vectors
        # from this model. Switching embedding models would invalidate the
        # whole index and require a re-build.
        self.default_embedding = os.environ.get(
            "GEMINI_MODEL_EMBEDDING", "gemini-embedding-001",
        )

        # Running totals (in-process, reset on restart)
        self.totals = LLMCallStats(model="<aggregate>", cache_hit=False)

    # ──────────────────────────────────────────────────────────────────────
    # generate(): one structured-output (or free-form) call
    # ──────────────────────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_schema: type[SchemaT] | None = None,
        temperature: float = 0.0,
        system_instruction: str | None = None,
        max_output_tokens: int | None = None,
        force_refresh: bool = False,
    ) -> tuple[Any, LLMCallStats]:
        """Generate a single response from Gemini.

        Returns
        -------
        (parsed, stats)
            parsed : an instance of `response_schema` if given, else the raw
                     response text (str).
            stats  : an LLMCallStats describing the call.
        """
        model = model or self.default_flash
        cache_key = self._build_cache_key(
            kind="generate",
            model=model,
            prompt=prompt,
            response_schema=response_schema,
            temperature=temperature,
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
        )

        # ── Cache hit path ────────────────────────────────────────────────
        if not self.cache_disabled and not force_refresh:
            cached = self._cache_get(cache_key)
            if cached is not None:
                stats = LLMCallStats(
                    model=model,
                    cache_hit=True,
                    prompt_tokens=cached.get("prompt_tokens", 0),
                    completion_tokens=cached.get("completion_tokens", 0),
                    total_tokens=cached.get("total_tokens", 0),
                    cost_usd=0.0,
                    cost_inr=0.0,
                    latency_seconds=0.0,
                    response_text_length=len(cached.get("response_text", "")),
                )
                if response_schema is not None:
                    parsed = response_schema.model_validate(cached["parsed_dict"])
                    return parsed, stats
                return cached["response_text"], stats

        # ── Live API call (with retry) ────────────────────────────────────
        t0 = time.monotonic()
        try:
            raw_response = self._call_api_with_retry(
                model=model,
                prompt=prompt,
                response_schema=response_schema,
                temperature=temperature,
                system_instruction=system_instruction,
                max_output_tokens=max_output_tokens,
            )
        except Exception as e:
            stats = LLMCallStats(model=model, cache_hit=False, error=str(e))
            logger.exception("Gemini generate() failed: %s", e)
            raise

        latency = time.monotonic() - t0

        # ── Parse usage metadata for cost tracking ────────────────────────
        usage = getattr(raw_response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", prompt_tokens + completion_tokens)
        cost_usd = _estimate_cost_usd(model, prompt_tokens, completion_tokens)

        # ── Extract response ──────────────────────────────────────────────
        response_text = raw_response.text if hasattr(raw_response, "text") else ""
        parsed_dict: dict[str, Any] | None = None
        parsed_obj: Any = response_text

        if response_schema is not None:
            # google-genai returns .parsed for structured output
            parsed_obj = getattr(raw_response, "parsed", None)
            if parsed_obj is None:
                # Fall back to manual JSON parse
                try:
                    parsed_dict = json.loads(response_text)
                    parsed_obj = response_schema.model_validate(parsed_dict)
                except Exception as e:
                    logger.error(
                        "Failed to parse structured response from %s. "
                        "Response text was: %s",
                        model,
                        response_text[:500],
                    )
                    raise ValueError(
                        f"Could not parse Gemini response as {response_schema.__name__}: {e}"
                    ) from e
            else:
                if isinstance(parsed_obj, BaseModel):
                    parsed_dict = parsed_obj.model_dump()
                elif isinstance(parsed_obj, dict):
                    parsed_dict = parsed_obj
                    parsed_obj = response_schema.model_validate(parsed_obj)
                elif isinstance(parsed_obj, list):
                    # Some schemas wrap a list — re-wrap into a Pydantic model
                    # if the schema is a list-typed root. Otherwise bail.
                    parsed_dict = {"__root__": parsed_obj}
                    parsed_obj = response_schema.model_validate(parsed_obj)

        # ── Cache write ───────────────────────────────────────────────────
        if not self.cache_disabled:
            self._cache_put(
                cache_key,
                {
                    "model": model,
                    "prompt": prompt,
                    "response_text": response_text,
                    "parsed_dict": parsed_dict,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": cost_usd,
                },
            )

        stats = LLMCallStats(
            model=model,
            cache_hit=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_inr=cost_usd * USD_TO_INR,
            latency_seconds=latency,
            response_text_length=len(response_text),
        )
        self._add_to_totals(stats)
        return parsed_obj, stats

    # ──────────────────────────────────────────────────────────────────────
    # embed(): produce vector embeddings for a list of texts
    # ──────────────────────────────────────────────────────────────────────
    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        force_refresh: bool = False,
    ) -> tuple[list[list[float]], LLMCallStats]:
        """Embed a list of texts. Returns (embeddings, stats).

        Caches by (model, batch-hash). Embeds in batches of 100 to respect
        per-call limits.
        """
        model = model or self.default_embedding
        cache_key = self._build_cache_key(kind="embed", model=model, prompt=json.dumps(texts))
        if not self.cache_disabled and not force_refresh:
            cached = self._cache_get(cache_key)
            if cached is not None:
                stats = LLMCallStats(
                    model=model,
                    cache_hit=True,
                    embedding_dim=len(cached["embeddings"][0]) if cached["embeddings"] else 0,
                    num_inputs=len(texts),
                )
                return cached["embeddings"], stats

        t0 = time.monotonic()
        # google-genai client.models.embed_content expects a string OR list of strings.
        # Paid tier has generous embed RPM; default batches of 50 with no sleep.
        # Free tier callers can shrink BATCH and add SLEEP if they hit 429s
        # (the retry decorator will surface the error if the rate is too high).
        all_embeds: list[list[float]] = []
        BATCH = int(os.environ.get("GEMINI_EMBED_BATCH_SIZE", "50"))
        SLEEP_BETWEEN_BATCHES = float(os.environ.get("GEMINI_EMBED_SLEEP_SECONDS", "0.0"))
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            resp = self._embed_with_retry(model=model, texts=batch)
            for e in resp.embeddings:
                # `e.values` is the List[float]
                all_embeds.append(list(e.values))
            if SLEEP_BETWEEN_BATCHES > 0 and i + BATCH < len(texts):
                time.sleep(SLEEP_BETWEEN_BATCHES)
        latency = time.monotonic() - t0

        # Cost (per char)
        total_chars = sum(len(t) for t in texts)
        per_1m = EMBEDDING_PRICING_USD_PER_1M_CHARS.get(model, 0.025)
        cost_usd = (total_chars / 1_000_000) * per_1m

        if not self.cache_disabled:
            self._cache_put(cache_key, {"embeddings": all_embeds, "model": model})

        stats = LLMCallStats(
            model=model,
            cache_hit=False,
            cost_usd=cost_usd,
            cost_inr=cost_usd * USD_TO_INR,
            latency_seconds=latency,
            embedding_dim=len(all_embeds[0]) if all_embeds else 0,
            num_inputs=len(texts),
        )
        self._add_to_totals(stats)
        return all_embeds, stats

    # ──────────────────────────────────────────────────────────────────────
    # Internal: API call with retry on transient errors
    # ──────────────────────────────────────────────────────────────────────
    @retry(
        retry=retry_if_exception_type(_GeminiTransientError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _call_api_with_retry(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: type[BaseModel] | None,
        temperature: float,
        system_instruction: str | None,
        max_output_tokens: int | None,
    ) -> Any:
        from google.genai import types as genai_types

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if max_output_tokens:
            config_kwargs["max_output_tokens"] = max_output_tokens
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        try:
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
            return response
        except Exception as e:
            msg = str(e).lower()
            if any(s in msg for s in ("429", "503", "500", "rate limit", "unavailable", "deadline")):
                raise _GeminiTransientError(str(e)) from e
            raise

    @retry(
        retry=retry_if_exception_type(_GeminiTransientError),
        stop=stop_after_attempt(8),
        # Embedding free tier is 100 RPM → can need 30s+ waits. Allow up to
        # 60s between attempts so a single 429 doesn't cause us to bail.
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _embed_with_retry(self, *, model: str, texts: list[str]) -> Any:
        try:
            return self._client.models.embed_content(model=model, contents=texts)
        except Exception as e:
            msg = str(e).lower()
            if any(s in msg for s in ("429", "503", "500", "rate limit", "unavailable", "deadline", "resource_exhausted")):
                raise _GeminiTransientError(str(e)) from e
            raise

    # ──────────────────────────────────────────────────────────────────────
    # Internal: cache key building + I/O
    # ──────────────────────────────────────────────────────────────────────
    def _build_cache_key(
        self,
        *,
        kind: str,
        model: str,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        system_instruction: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        schema_repr = ""
        if response_schema is not None:
            try:
                schema_repr = json.dumps(
                    response_schema.model_json_schema(), sort_keys=True
                )
            except Exception:
                schema_repr = response_schema.__name__

        payload = json.dumps(
            {
                "kind": kind,
                "model": model,
                "prompt": prompt,
                "schema": schema_repr,
                "temp": temperature,
                "sys": system_instruction or "",
                "max_out": max_output_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        # Shard into subdirs by first 2 chars to avoid one giant directory
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Cache read failed for %s; ignoring.", key[:8])
            return None

    def _cache_put(self, key: str, value: dict[str, Any]) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, default=_json_default)
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", key[:8], e)

    # ──────────────────────────────────────────────────────────────────────
    # Internal: aggregate totals
    # ──────────────────────────────────────────────────────────────────────
    def _add_to_totals(self, stats: LLMCallStats) -> None:
        if stats.cache_hit:
            return
        self.totals.prompt_tokens += stats.prompt_tokens
        self.totals.completion_tokens += stats.completion_tokens
        self.totals.total_tokens += stats.total_tokens
        self.totals.cost_usd += stats.cost_usd
        self.totals.cost_inr += stats.cost_inr
        self.totals.latency_seconds += stats.latency_seconds
        self.totals.response_text_length += stats.response_text_length

    def reset_totals(self) -> None:
        self.totals = LLMCallStats(model="<aggregate>", cache_hit=False)


# ─── Module-level helpers ───────────────────────────────────────────────────
_default_client: GeminiClient | None = None


def get_default_client():
    """Return the process-wide shared LLM client.

    If `GROQ_API_KEY` or `GEMINI_API_KEY_OLD` is set, returns a
    `FallbackLLMClient` that transparently fails over from the primary
    Gemini key to Groq (LLM only) to a second Gemini key on quota errors.
    Otherwise returns a plain `GeminiClient`.

    The wrapper has the same `.generate(...)`, `.embed(...)`, and
    `.default_*` API surface as `GeminiClient` so call sites don't need
    to change.
    """
    has_fallback = bool(
        os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY_OLD")
    )
    if has_fallback:
        try:
            from src.llm.fallback import get_fallback_client
            return get_fallback_client()
        except Exception as e:
            logger.warning(
                "Fallback client init failed; falling back to plain Gemini: %s", e
            )
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client


def _project_root() -> Path:
    """Walk up from this file to find the project root (the dir containing
    pyproject.toml). Falls back to cwd if not found."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICING_USD_PER_1M_TOKENS.get(model)
    if not pricing:
        # Try a fuzzy match for newer / legacy aliases
        for k, v in PRICING_USD_PER_1M_TOKENS.items():
            if k in model or model in k:
                pricing = v
                break
    if not pricing:
        return 0.0
    return (
        (prompt_tokens / 1_000_000) * pricing["input"]
        + (completion_tokens / 1_000_000) * pricing["output"]
    )


def _json_default(o: Any) -> Any:
    """Fallback JSON serialiser for cache writes (handles Pydantic + dataclasses)."""
    if isinstance(o, BaseModel):
        return o.model_dump()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


# ─── Field-level shim for older test imports ────────────────────────────────
# Allows `from src.llm import gemini_client; gemini_client.GeminiClient(...)`.
__all__ = [
    "GeminiClient",
    "LLMCallStats",
    "get_default_client",
    "PRICING_USD_PER_1M_TOKENS",
    "EMBEDDING_PRICING_USD_PER_1M_CHARS",
    "USD_TO_INR",
]


# Marker to silence `field` import warning when dataclasses isn't fully used.
_ = field
