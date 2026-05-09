"""Quick eyeball-review summary for JK_001 pipeline output.

Usage: python scripts/jk_001_summary.py
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

CORPUS = Path("corpus/JK_001/parsed")


def _read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    flags    = _read_jsonl(CORPUS / "candidate_flags.jsonl")
    verdicts = _read_jsonl(CORPUS / "stage2_verdicts.jsonl")
    confirmed= _read_jsonl(CORPUS / "stage3_confirmed.jsonl")
    rewrites = _read_jsonl(CORPUS / "stage4_rewrites.jsonl")

    print("# JK_001 -- pipeline summary\n")

    # Stage 1
    n_total_flags = len(flags)
    n_real_flags = sum(1 for f in flags if f.get("flagged"))
    print("Stage 1 BCT:")
    print(f"  candidate flags emitted: {n_total_flags}")
    print(f"  flagged (>=1 CANNOT_DETERMINE): {n_real_flags}")
    if flags:
        reasons = Counter(f.get("flag_reason", "?") for f in flags if f.get("flagged"))
        for r, c in reasons.most_common(5):
            print(f"    {r}: {c}")

    # Stage 2
    print("\nStage 2 critic:")
    if verdicts:
        verdict_counts = Counter(v.get("verdict", "?") for v in verdicts)
        for v, c in verdict_counts.most_common():
            print(f"  {v}: {c}")
    else:
        print("  (not yet run)")

    # Stage 3
    print("\nStage 3 confirm + severity:")
    if confirmed:
        critique_counts = Counter(c.get("critique_verdict", "?") for c in confirmed)
        for v, c in critique_counts.most_common():
            print(f"  {v}: {c}")
        sev_counts = Counter(c.get("severity_tier", "?") for c in confirmed
                              if c.get("critique_verdict") == "CONFIRMED")
        print("  severity (confirmed only):")
        for s in ("high", "medium", "low"):
            print(f"    {s}: {sev_counts.get(s, 0)}")
    else:
        print("  (not yet run)")

    # Stage 4
    print("\nStage 4 rewrite:")
    if rewrites:
        status_counts = Counter(r.get("status", "?") for r in rewrites)
        for s, c in status_counts.most_common():
            print(f"  {s}: {c}")
        accepted = [r for r in rewrites if r.get("status") == "ACCEPTED"]
        if accepted:
            avg_cosine = sum(r.get("intent_cosine", 0) for r in accepted) / len(accepted)
            avg_attempts = sum(r.get("attempts", 0) for r in accepted) / len(accepted)
            print(f"  avg attempts (accepted): {avg_attempts:.2f}")
            print(f"  avg intent cosine (accepted): {avg_cosine:.3f}")
    else:
        print("  (not yet run)")

    # Sample 3 confirmed-high rewrites for eyeball review
    if rewrites:
        print("\n# Sample rewrites for eyeball review:\n")
        for r in rewrites[:3]:
            print(f"--- {r['flag_id']} ({r.get('status')}, attempts={r.get('attempts')}) ---")
            print(f"ORIGINAL: {r.get('original_text','')[:200]}")
            print(f"REWRITE : {r.get('rewrite_text','')[:300]}")
            print(f"CD: {r.get('original_cd_count')} -> {r.get('rewrite_cd_count')}, "
                  f"intent={r.get('intent_cosine')}")
            print()


if __name__ == "__main__":
    main()
