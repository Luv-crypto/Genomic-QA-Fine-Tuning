#!/usr/bin/env python
"""Tally pairwise A/B feedback and suggest promotion.

Usage: python Scripts/ab_tally.py <candidate_id> [--hours 24] [--threshold 0.5]
"""
import json, argparse, datetime as dt
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Check pairwise votes")
    ap.add_argument("candidate_id", help="Internal model id of candidate")
    ap.add_argument("--feedback-file", default="user_feedback.json")
    ap.add_argument("--hours", type=float, default=24.0, help="Lookback window")
    ap.add_argument("--threshold", type=float, default=0.5, help="Promotion threshold")
    args = ap.parse_args()

    path = Path(args.feedback_file)
    if not path.exists():
        print(f"No feedback file found: {path}")
        return

    data = json.loads(path.read_text())
    cutoff = dt.datetime.utcnow() - dt.timedelta(hours=args.hours)

    total = 0
    cand = 0
    for row in data.get("pair", []):
        ts = row.get("timestamp")
        if not ts:
            continue
        try:
            t = dt.datetime.fromisoformat(ts)
        except ValueError:
            continue
        if t < cutoff:
            continue
        pref = row.get("preferred")
        if pref not in {"A", "B"}:
            continue
        a_id = row.get("a_model")
        b_id = row.get("b_model")
        if pref == "A" and a_id == args.candidate_id:
            cand += 1
        elif pref == "B" and b_id == args.candidate_id:
            cand += 1
        total += 1

    if total == 0:
        print("No votes found in window")
        return

    ratio = cand / total
    print(f"Candidate {args.candidate_id}: {cand}/{total} votes ({ratio:.1%})")
    if ratio >= args.threshold:
        print("PROMOTE")
    else:
        print("REJECT")

if __name__ == "__main__":
    main()
