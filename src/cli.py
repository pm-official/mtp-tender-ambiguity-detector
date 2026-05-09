"""Tender Ambiguity Detector — top-level CLI.

Subcommands grow as the pipeline does. Today: only `verify` (Session 1).
Tomorrow: `acquire`, `parse`, `score`, `run` will be added by later sessions.

Usage:
    python -m src.cli verify          # one cheap Gemini call to confirm setup
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from pydantic import BaseModel, Field


def cmd_verify(args: argparse.Namespace) -> int:
    """Make a tiny live Gemini call to confirm the API key, the SDK, and the
    cache wrapper all work. Reports tokens, cost, and cache behaviour."""
    from src.llm.gemini_client import GeminiClient

    class _Reply(BaseModel):
        greeting: str = Field(..., description="A short friendly greeting")
        confidence: float = Field(..., ge=0.0, le=1.0)

    print("Tender Ambiguity Detector -- installation verifier")
    print("-" * 60)

    try:
        client = GeminiClient()
    except RuntimeError as e:
        print(f"\n  [FAIL]  {e}")
        print(
            "\n     Quick fix:"
            "\n       1.  copy .env.example .env"
            "\n       2.  open .env in a text editor"
            "\n       3.  replace YOUR_GEMINI_API_KEY_HERE with your real key"
            "\n           from https://aistudio.google.com/app/apikey"
        )
        return 1

    print("  [OK]   GeminiClient constructed (API key found)")
    print(f"  ..    cache dir: {client.cache_dir}")
    print(f"  ..    default flash model: {client.default_flash}")
    print(f"  ..    default pro model:   {client.default_pro}")
    print(f"  ..    default embedding:   {client.default_embedding}")

    # Live structured-output call
    print("\n  -> calling Gemini 2.5 Flash with a 1-line prompt...")
    try:
        parsed, stats = client.generate(
            prompt=(
                "Return a JSON object with two fields: greeting (a short "
                "friendly hello), and confidence (a float between 0 and 1)."
            ),
            response_schema=_Reply,
            model=client.default_flash,
            temperature=0.0,
            force_refresh=args.fresh,
        )
    except Exception as e:
        print(f"\n  [FAIL]  Live call failed: {e}")
        print(
            "\n     If the error mentions invalid API key, double-check the"
            "\n     value in .env. If it mentions billing or quota, check"
            "\n     https://aistudio.google.com/ for usage limits."
        )
        return 2

    print(f"  [OK]   parsed response:  {parsed!r}")
    print(f"  ..    cache hit: {stats.cache_hit}")
    print(f"  ..    tokens in/out/total: "
          f"{stats.prompt_tokens} / {stats.completion_tokens} / {stats.total_tokens}")
    print(f"  ..    cost: USD {stats.cost_usd:.6f}  (~ INR {stats.cost_inr:.4f})")
    print(f"  ..    latency: {stats.latency_seconds:.2f} s")

    # Repeat to confirm cache works
    print("\n  -> repeating identical call to test cache...")
    parsed2, stats2 = client.generate(
        prompt=(
            "Return a JSON object with two fields: greeting (a short "
            "friendly hello), and confidence (a float between 0 and 1)."
        ),
        response_schema=_Reply,
        model=client.default_flash,
        temperature=0.0,
    )
    if stats2.cache_hit:
        print("  [OK]   cache HIT -- repeat call cost USD 0.000000  (free)")
    else:
        print("  [WARN] cache MISS on identical call -- cache may not be writable. "
              "Check disk permissions on .cache/")

    print("\n" + "-" * 60)
    print("  All good. Proceed to Session 2 (tender + corrigendum acquisition).")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tender-ambiguity-detector")
    sub = p.add_subparsers(dest="command", required=True)

    verify = sub.add_parser(
        "verify",
        help="Make a small live Gemini call to confirm the install works.",
    )
    verify.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the cache and force a fresh API call.",
    )
    verify.set_defaults(func=cmd_verify)

    # Session 2 — acquire sub-commands (init / drop / validate / synthesize / scrape / list)
    from src.acquire.cli import add_subparser as add_acquire_subparser
    add_acquire_subparser(sub)

    # Session 3 — parse sub-commands (run / inspect)
    from src.parse.cli import add_subparser as add_parse_subparser
    add_parse_subparser(sub)

    # Session 4 — extract sub-commands (run / query / summary)
    from src.extract.cli import add_subparser as add_extract_subparser
    add_extract_subparser(sub)

    # Session 6 — score sub-commands (run / inspect) — Stage 1 BCT
    from src.score.cli import add_subparser as add_score_subparser
    add_score_subparser(sub)

    # Session 7 — critique sub-commands (run / inspect) — Stage 2 apparent-vs-real
    from src.critic.cli import add_subparser as add_critique_subparser
    add_critique_subparser(sub)

    # Session 8 — confirm sub-commands (run / inspect) — Stage 3 critique + severity
    from src.critic.cli_stage3 import add_subparser as add_confirm_subparser
    add_confirm_subparser(sub)

    # Feature B — pipeline sub-commands (run / status / history)
    from src.pipeline_cli import add_subparser as add_pipeline_subparser
    add_pipeline_subparser(sub)

    # Feature C — report sub-commands (html)
    from src.report.cli import add_subparser as add_report_subparser
    add_report_subparser(sub)

    # Session 9 — iscode sub-commands (build / query / verify / summary)
    from src.iscode.cli import add_subparser as add_iscode_subparser
    add_iscode_subparser(sub)

    # Session 9 — rewrite sub-commands (run / inspect) — Stage 4
    from src.rewrite.cli import add_subparser as add_rewrite_subparser
    add_rewrite_subparser(sub)

    # Session 10 — eval sub-commands (lexicon / metrics / compare)
    from src.eval.cli import add_subparser as add_eval_subparser
    add_eval_subparser(sub)

    return p


def main(argv: list[str] | None = None) -> NoReturn:
    parser = _build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
