"""
Retry Failed Extractions
-------------------------
Finds rows with "error": "extraction_failed" in an existing .jsonl output file,
re-runs the Gemini API call for each one, and overwrites the null vectors in place.

Usage:
    python retry_failed.py <jsonl_file> [--csv <original_csv>]

If --csv is not provided, the script retries using only the text already stored
in the jsonl (which is enough — the full tweet body is saved there).

Requirements:
    pip install google-genai
    PROJECT_ID must be set below (same as hydro_feature_extractor.py)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from google import genai

# ── Configuration (must match hydro_feature_extractor.py) ────────────────────

MODEL      = "gemini-2.5-flash"
PROJECT_ID = "still-entity-494814-m3"
LOCATION   = "us-central1"

MAX_TOKENS       = 1024
MAX_ATTEMPTS     = 5
BASE_RETRY_DELAY = 15
REQUEST_DELAY    = 1.5

SYSTEM_PROMPT = """Act as a specialized hydrological data engineer. Your task is to perform zero-shot semantic feature extraction on tweets to detect the onset of floods or landslides. For the input text, output a JSON object containing a 'feature_vector' with the following 6 numeric dimensions and 1 categorical field.

1. immediacy: 1.0 if the event is happening 'right now' or 'minutes ago'; 0.0 if it is a news recap of a past event.
2. first_hand_witness: 1.0 if the user is describing what they see or feel; 0.0 if they are sharing a link or quoting others.
3. locational_precision: 1.0 if specific local landmarks or street names are mentioned; 0.0 if only a city or country is named.
4. severity_index: 1.0 if life-threatening damage is mentioned; 0.1 if it's just 'heavy rain'.
5. distress_signal: 1.0 if there is a linguistic markers of panic or emergency; 0.0 if purely informative.
6. hydro_relevance: 1.0 if the tweet is definitely about water/land displacement; 0.0 if it is metaphorical (e.g., 'a flood of emails').
7. event_type: one of "flood", "landslide", or null. Set to "flood" if the tweet is clearly about flooding or inundation. Set to "landslide" if clearly about a landslide, mudslide, or ground displacement. Set to null if the tweet is not clearly about either (e.g. drought, unrelated content, ambiguous).

Output a JSON object with a single key 'feature_vector' containing all 7 fields. Nothing else."""

# ── API call ─────────────────────────────────────────────────────────────────

def extract_features(client, prompt: str) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_TOKENS,
                    temperature=0.0,
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [warn] JSON parse error on attempt {attempt}/{MAX_ATTEMPTS}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  [warn] API error on attempt {attempt}/{MAX_ATTEMPTS}: {e}", file=sys.stderr)
            delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_ATTEMPTS:
                print(f"  [info] Waiting {delay}s before retry ...", file=sys.stderr)
                time.sleep(delay)
    return None  # signal failure

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retry failed rows in a hydro feature JSONL file.")
    parser.add_argument("jsonl", help="Path to the .jsonl output file")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        sys.exit(f"Error: file not found: {jsonl_path}")

    # Load all records
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    failed = [(i, r) for i, r in enumerate(records) if r.get("error") == "extraction_failed"]
    print(f"Total records : {len(records)}")
    print(f"Failed rows   : {len(failed)}")

    if not failed:
        print("Nothing to retry!")
        return

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    succeeded = 0
    still_failed = 0

    for idx, (record_idx, record) in enumerate(failed, start=1):
        tweet_id = record.get("tweet_id", f"row_{record_idx}")
        body = record.get("text", "").strip()

        if not body:
            print(f"  [{idx}/{len(failed)}] Skipping — no text stored for tweet {tweet_id}")
            still_failed += 1
            continue

        print(f"  [{idx}/{len(failed)}] Retrying tweet {tweet_id} ...")
        time.sleep(REQUEST_DELAY)

        # Build the same compact prompt as the main extractor
        prompt = json.dumps({
            "tweet_id":     tweet_id,
            "text":         body,
            "language":     record.get("language", ""),
            "country":      record.get("country", ""),
            "published_at": record.get("published_at", ""),
            "keyword":      record.get("keyword", ""),
        }, ensure_ascii=False)

        features = extract_features(client, prompt)

        if features:
            # Overwrite the failed record in place
            records[record_idx].update(features)
            records[record_idx].pop("error", None)
            succeeded += 1
            print(f"  [{idx}/{len(failed)}] ✓ Success")
        else:
            still_failed += 1
            print(f"  [{idx}/{len(failed)}] ✗ Still failing — leaving null vector")

    # Write all records back
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. {succeeded} recovered | {still_failed} still failed")
    print(f"Output written back to '{jsonl_path}'")


if __name__ == "__main__":
    main()
