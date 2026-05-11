"""
Hydrological Feature Extractor (Integrated Version)
--------------------------------------------------
Combines the complete processing logic from hydro_feature_extractor.py 
with the threaded timeout and rate-limit handling from hydro_features_ex2.py.
"""

import argparse
import csv
import json
import os
import sys
import time
import concurrent.futures
from pathlib import Path

from google import genai

# ── Configuration ────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash"
PROJECT_ID = "still-entity-494814-m3"   #
LOCATION  = "us-central1"               #
MAX_TOKENS = 1024
MAX_ATTEMPTS = 5           
BASE_RETRY_DELAY = 5       
REQUEST_TIMEOUT = 60       # Timeout enforced via threading

# Taken from hydro_feature_extractor.py
SYSTEM_PROMPT = """Act as a specialized hydrological data engineer. Your task is to perform zero-shot semantic feature extraction on tweets to detect the onset of floods or landslides. For the input text, output a JSON object containing a 'feature_vector' with the following 6 numeric dimensions and 1 categorical field.

1. immediacy: 1.0 if the event is happening 'right now' or 'minutes ago'; 0.0 if it is a news recap of a past event.
2. first_hand_witness: 1.0 if the user is describing what they see or feel; 0.0 if they are sharing a link or quoting others.
3. locational_precision: 1.0 if specific local landmarks or street names are mentioned; 0.0 if only a city or country is named.
4. severity_index: 1.0 if life-threatening damage is mentioned; 0.1 if it's just 'heavy rain'.
5. distress_signal: 1.0 if there is a linguistic markers of panic or emergency; 0.0 if purely informative.
6. hydro_relevance: 1.0 if the tweet is definitely about water/land displacement; 0.0 if it is metaphorical (e.g., 'a flood of emails').
7. event_type: one of "flood", "landslide", or null. Set to "flood" if the tweet is clearly about flooding or inundation. Set to "landslide" if clearly about a landslide, mudslide, or ground displacement. Set to null if the tweet is not clearly about either (e.g. drought, unrelated content, ambiguous).

Output a JSON object with a single key 'feature_vector' containing all 7 fields. Nothing else."""

NULL_VECTOR = {
    "feature_vector": {
        "immediacy": None,
        "first_hand_witness": None,
        "locational_precision": None,
        "severity_index": None,
        "distress_signal": None,
        "hydro_relevance": None,
        "event_type": None,
    },
    "error": "extraction_failed",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def interruptible_sleep(seconds: float) -> None:
    """Sleep logic that allows Ctrl+C on Windows."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(0.5)

def build_prompt(row: dict) -> str:
    """Constructs the prompt payload from CSV row."""
    payload = {
        "tweet_id":     row.get("tweet_id", row.get("_id", "")),
        "text":         (row.get("body") or "").strip(),
        "language":     row.get("lang", ""),
        "country":      row.get("country_name", row.get("location.country.label.eng", "")),
        "published_at": row.get("dateTimePub", row.get("dateTime", "")),
        "keyword":      row.get("keyword", ""),
    }
    return json.dumps(payload, ensure_ascii=False)

def load_completed_ids(output_path: Path) -> set:
    """Enables resume behavior by checking the output file."""
    done = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line.strip())
                    tid = rec.get("tweet_id")
                    if tid: done.add(str(tid))
                except json.JSONDecodeError: pass
    return done

# ── core extraction logic ────────────────────────────────────────────────────

def extract_features(client: genai.Client, prompt: str) -> dict:
    """
    Improved extraction using ThreadPoolExecutor for timeouts and 
    handling 429/503 status codes.
    """
    delay = BASE_RETRY_DELAY
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # Use threading to enforce the REQUEST_TIMEOUT
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    client.models.generate_content,
                    model=MODEL,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=MAX_TOKENS,
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                response = future.result(timeout=REQUEST_TIMEOUT)
            
            return json.loads(response.text.strip())

        except concurrent.futures.TimeoutError:
            print(f"  [warn] Timed out on attempt {attempt}/{MAX_ATTEMPTS}", file=sys.stderr)

        except json.JSONDecodeError as e:
            print(f"  [warn] JSON parse error on attempt {attempt}: {e}", file=sys.stderr)

        except KeyboardInterrupt:
            raise

        except Exception as e:
            msg = str(e)
            print(f"  [warn] API error on attempt {attempt}: {msg}", file=sys.stderr)
            # Expanded retry logic to include 429 (Rate Limit)
            if any(x in msg for x in ("503", "429", "UNAVAILABLE")) and attempt < MAX_ATTEMPTS:
                print(f"  [info] Waiting {delay}s before retry ...", file=sys.stderr)
                interruptible_sleep(delay)
                delay = min(delay * 2, 60)
                continue

        if attempt < MAX_ATTEMPTS:
            interruptible_sleep(2)

    return NULL_VECTOR

# ── Main Loop ────────────────────────────────────────────────────────────────

def process_file(input_path: str, output_path: str) -> None:
    """Orchestrates the CSV reading and JSONL writing."""
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_ids(output_path)
    if completed:
        print(f"Resuming — {len(completed)} tweets already processed.")

    written = skipped = failed = 0

    try:
        with (
            input_path.open(encoding="utf-8", newline="") as fh,
            output_path.open("a", encoding="utf-8") as out,
        ):
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader, start=1):
                tweet_id = str(row.get("tweet_id", row.get("_id", f"row_{i}")))
                body = (row.get("body") or "").strip()

                if not body or tweet_id in completed:
                    skipped += 1
                    continue

                print(f"  [{i}] Extracting features for tweet {tweet_id} ...")
                features = extract_features(client, build_prompt(row))

                result = {
                    "tweet_id":     tweet_id,
                    "text":         body,
                    "language":     row.get("lang", ""),
                    "country":      row.get("country_name", ""),
                    "published_at": row.get("dateTimePub", row.get("dateTime", "")),
                    "keyword":      row.get("keyword", ""),
                    **features,
                }
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()

                if features.get("error"): failed += 1
                else: written += 1

    except KeyboardInterrupt:
        print("\n[interrupted] Stopping gracefully...")
        sys.exit(0)

    print(f"\nDone. {written} succeeded | {failed} failed | {skipped} skipped.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated Hydrological Feature Extractor.")
    parser.add_argument("input", help="Path to the input CSV/TXT file")
    parser.add_argument("--output", default=None, help="Path to the output JSONL file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output = args.output or str(Path(args.input).with_suffix("")) + "_features.jsonl"
    process_file(args.input, output)