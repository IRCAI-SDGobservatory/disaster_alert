"""
Annotation Sampler v2.0
------------------------
Stratified sampling of tweets for expert annotation using ABSOLUTE score
thresholds, so zone boundaries are identical across all case study files.
A tweet with score 0.4 is always HIGH whether it comes from Germany, Belgium,
Indonesia, or Japan.

Zone thresholds (tune in configuration block below):
  HIGH   score >= 0.30   → clearly disaster-relevant
  MIDDLE 0.05 <= score < 0.30   → ambiguous
  LOW    score <  0.05   → clearly irrelevant

Sampling targets (50 tweets total):
  20 HIGH + 10 MIDDLE + 20 LOW

If a zone has fewer tweets than the target, all tweets from that zone are
taken and a warning is printed (rather than silently falling back to relative
zones, which would make cross-file comparison invalid).

Produces two CSV files from the same 50-tweet sample:
  <stem>_annotate_blind.csv   — text only (for unbiased labelling)
  <stem>_annotate_full.csv    — text + all feature scores (for comparison)

Usage:
    python annotation_sampler.py <jsonl_file> [--seed 42] [--stats]

    --stats   Print score distribution and zone counts, then exit.
              Run this first on each new file to verify thresholds make sense.

Examples:
    python annotation_sampler.py GermanyJuly2021_features.jsonl --stats
    python annotation_sampler.py GermanyJuly2021_features.jsonl
    python annotation_sampler.py BelgiumJuly2021_features.jsonl --seed 42
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Absolute zone thresholds — identical across ALL case study files
HIGH_THRESHOLD   = 0.30   # score >= this → HIGH
MIDDLE_THRESHOLD = 0.05   # score >= this and < HIGH_THRESHOLD → MIDDLE
                           # score <  this → LOW

# Sampling targets per zone
N_HIGH   = 20
N_MIDDLE = 10
N_LOW    = 20

# Boost weights — must match alert_engine.py
BOOST_WEIGHTS = {
    "immediacy":            0.40,
    "distress_signal":      0.50,
    "first_hand_witness":   0.30,
    "locational_precision": 0.20,
}

# ── Composite score ───────────────────────────────────────────────────────────

def composite_score(fv: dict) -> float:
    hr   = float(fv.get("hydro_relevance") or 0.0)
    si   = float(fv.get("severity_index")  or 0.0)
    gate = hr * si
    boost = 1.0 + sum(
        w * float(fv.get(k) or 0.0)
        for k, w in BOOST_WEIGHTS.items()
    )
    return gate * boost

# ── Load ──────────────────────────────────────────────────────────────────────

def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    skipped = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error") == "extraction_failed":
                skipped += 1
                continue
            fv = rec.get("feature_vector", {})
            if not fv:
                skipped += 1
                continue
            rec["_score"] = composite_score(fv)
            records.append(rec)
    if skipped:
        print(f"  [info] Skipped {skipped} records (failed extraction or missing feature vector)")
    return records

# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(records: list[dict], jsonl_path: Path) -> None:
    scores = sorted(r["_score"] for r in records)
    n = len(scores)

    high   = [r for r in records if r["_score"] >= HIGH_THRESHOLD]
    middle = [r for r in records if MIDDLE_THRESHOLD <= r["_score"] < HIGH_THRESHOLD]
    low    = [r for r in records if r["_score"] < MIDDLE_THRESHOLD]

    print(f"\n── Score distribution: {jsonl_path.name} ──")
    print(f"  Total valid records : {n}")
    print(f"\n  Percentiles:")
    for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        idx = min(int(p / 100 * n), n - 1)
        print(f"    p{p:3d} : {scores[idx]:.4f}")
    print(f"\n  Zone counts (absolute thresholds):")
    print(f"    HIGH   (score >= {HIGH_THRESHOLD})                    : {len(high):5d}  ({100*len(high)/n:.1f}%)")
    print(f"    MIDDLE ({MIDDLE_THRESHOLD} <= score < {HIGH_THRESHOLD}) : {len(middle):5d}  ({100*len(middle)/n:.1f}%)")
    print(f"    LOW    (score <  {MIDDLE_THRESHOLD})                    : {len(low):5d}  ({100*len(low)/n:.1f}%)")
    print(f"\n  Sampling targets: {N_HIGH} HIGH + {N_MIDDLE} MIDDLE + {N_LOW} LOW = {N_HIGH+N_MIDDLE+N_LOW} total")

    warns = []
    if len(high)   < N_HIGH:   warns.append(f"  WARNING: HIGH zone has only {len(high)} tweets (need {N_HIGH}) — all will be taken")
    if len(middle) < N_MIDDLE: warns.append(f"  WARNING: MIDDLE zone has only {len(middle)} tweets (need {N_MIDDLE}) — all will be taken")
    if len(low)    < N_LOW:    warns.append(f"  WARNING: LOW zone has only {len(low)} tweets (need {N_LOW}) — all will be taken")
    if warns:
        print()
        for w in warns: print(w)
    else:
        print(f"  All zones have sufficient tweets for target sample sizes.")

# ── Stratified sample ─────────────────────────────────────────────────────────

def stratified_sample(records: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)

    high   = [r for r in records if r["_score"] >= HIGH_THRESHOLD]
    middle = [r for r in records if MIDDLE_THRESHOLD <= r["_score"] < HIGH_THRESHOLD]
    low    = [r for r in records if r["_score"] < MIDDLE_THRESHOLD]

    sample_high   = rng.sample(high,   min(N_HIGH,   len(high)))
    sample_middle = rng.sample(middle, min(N_MIDDLE, len(middle)))
    sample_low    = rng.sample(low,    min(N_LOW,    len(low)))

    for r in sample_high:   r["_zone"] = "HIGH"
    for r in sample_middle: r["_zone"] = "MIDDLE"
    for r in sample_low:    r["_zone"] = "LOW"

    combined = sample_high + sample_middle + sample_low
    rng.shuffle(combined)   # hide zone order from annotator

    print(f"  Sampled: {len(sample_high)} HIGH + {len(sample_middle)} MIDDLE + {len(sample_low)} LOW = {len(combined)} total")
    return combined

# ── Write CSVs ────────────────────────────────────────────────────────────────

BLIND_COLS = [
    "row_num", "tweet_id", "published_at", "language",
    "country", "keyword", "text", "expert_label", "notes",
]

FULL_COLS = BLIND_COLS + [
    "zone", "composite_score", "hydro_relevance", "severity_index",
    "immediacy", "first_hand_witness", "locational_precision",
    "distress_signal", "event_type",
]

def write_blind(sample: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BLIND_COLS)
        writer.writeheader()
        for i, rec in enumerate(sample, start=1):
            writer.writerow({
                "row_num": i, "tweet_id": rec.get("tweet_id", ""),
                "published_at": rec.get("published_at", ""),
                "language": rec.get("language", ""), "country": rec.get("country", ""),
                "keyword": rec.get("keyword", ""), "text": rec.get("text", ""),
                "expert_label": "", "notes": "",
            })

def write_full(sample: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FULL_COLS)
        writer.writeheader()
        for i, rec in enumerate(sample, start=1):
            fv = rec.get("feature_vector", {})
            writer.writerow({
                "row_num": i, "tweet_id": rec.get("tweet_id", ""),
                "published_at": rec.get("published_at", ""),
                "language": rec.get("language", ""), "country": rec.get("country", ""),
                "keyword": rec.get("keyword", ""), "text": rec.get("text", ""),
                "expert_label": "", "notes": "",
                "zone": rec.get("_zone", ""),
                "composite_score": f"{rec['_score']:.4f}",
                "hydro_relevance": fv.get("hydro_relevance", ""),
                "severity_index": fv.get("severity_index", ""),
                "immediacy": fv.get("immediacy", ""),
                "first_hand_witness": fv.get("first_hand_witness", ""),
                "locational_precision": fv.get("locational_precision", ""),
                "distress_signal": fv.get("distress_signal", ""),
                "event_type": fv.get("event_type", ""),
            })

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Annotation sampler v2.0 — absolute score thresholds.")
    parser.add_argument("jsonl", help="Path to the _features.jsonl file")
    parser.add_argument("--seed",  type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--stats", action="store_true",
                        help="Print score distribution and zone counts only, then exit")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        sys.exit(f"File not found: {jsonl_path}")

    print(f"Loading '{jsonl_path}' ...")
    records = load_records(jsonl_path)
    print(f"  {len(records)} valid records loaded")

    print_stats(records, jsonl_path)

    if args.stats:
        return

    print(f"\nStratified sampling (seed={args.seed}) ...")
    sample = stratified_sample(records, seed=args.seed)

    stem       = jsonl_path.stem
    blind_path = jsonl_path.parent / f"{stem}_annotate_blind.csv"
    full_path  = jsonl_path.parent / f"{stem}_annotate_full.csv"

    write_blind(sample, blind_path)
    write_full(sample, full_path)

    print(f"\nOutput files:")
    print(f"  Blind (for annotation) : {blind_path}")
    print(f"  Full  (for comparison) : {full_path}")
    print(f"\nInstructions:")
    print(f"  1. Fill 'expert_label' in BLIND file: 1 = disaster-relevant, 0 = not")
    print(f"  2. Compare against FULL file once labelling is complete")
    print(f"  3. Match rows using row_num and tweet_id")
    print(f"  Thresholds: HIGH>={HIGH_THRESHOLD}, MIDDLE>={MIDDLE_THRESHOLD}, LOW<{MIDDLE_THRESHOLD}")

if __name__ == "__main__":
    main()
