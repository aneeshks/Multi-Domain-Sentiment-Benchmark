#!/usr/bin/env python3
"""
run_gemini_batch.py
══════════════════════════════════════════════════════════════════════
Submit all 8,872 rows to the Gemini Batch API (gemini-2.5-pro).
- Bypasses all RPM/RPD rate limits
- 50% cheaper than live API calls
- Results ready within 24 hours

Usage:
    python run_gemini_batch.py            # submit batch job
    python run_gemini_batch.py --poll     # check status + write results

Requirements:
    pip install google-genai python-dotenv pandas
    GEMINI_API_KEY in .env
"""

import os, re, sys, time, json
from pathlib import Path
from datetime import datetime

import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL      = "gemini-2.5-pro"
COL        = "pred_gemini25pro"
SAMPLE_DIR = Path("data/sampled")
DATASETS   = ["FPB", "SST2", "TweetEval", "TFNS", "VADER"]
JOB_FILE   = Path("data/gemini_batch_job_id.txt")

SYSTEM_PROMPT = (
    "You are a sentiment classifier. Classify the sentiment of the following "
    "text using these definitions:\n"
    "  positive — the text expresses optimism, approval, or good news\n"
    "  negative — the text expresses pessimism, criticism, or bad news\n"
    "  neutral  — the text is factual, balanced, or does not lean either way\n"
    "Reply with exactly one word: positive, negative, or neutral."
)

SYNONYMS = {
    "positive": {"positive", "bullish", "optimistic", "favorable", "favourable"},
    "negative": {"negative", "bearish", "pessimistic", "unfavorable", "unfavourable"},
    "neutral":  {"neutral", "factual", "balanced", "mixed"},
}

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─── Helpers ──────────────────────────────────────────────────────────────────
def normalise(raw: str) -> str:
    if not raw:
        return "UNPARSED"
    for tok in re.findall(r"[a-z]+", raw.lower()):
        for label, words in SYNONYMS.items():
            if tok in words:
                return label
    return "UNPARSED"

def build_requests() -> tuple[list, list]:
    """Build InlinedRequest list and matching metadata list."""
    requests = []
    metadata = []  # (dataset_name, row_index)

    for name in DATASETS:
        path = SAMPLE_DIR / f"sample_{name}.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if COL not in df.columns:
            df[COL] = ""

        pending = df.index[df[COL] == ""].tolist()
        print(f"  {name:<12} {len(pending)} rows to submit (skipping {len(df)-len(pending)} already done)")

        for idx in pending:
            sentence = df.at[idx, "sentence"]
            req = types.InlinedRequest(
                contents=f'Text: "{sentence}"\nSentiment:',
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=2000,
                ),
            )
            requests.append(req)
            metadata.append({"dataset": name, "index": int(idx)})

    return requests, metadata

def submit_batch():
    print("═" * 65)
    print(f"Gemini Batch API — {MODEL}")
    print(f"  Building requests...")
    print("═" * 65)

    requests, metadata = build_requests()
    if not requests:
        print("Nothing to submit — all rows already complete.")
        return

    print(f"\n  Total requests: {len(requests)}")
    print(f"  Submitting to Batch API...")

    batch_job = client.batches.create(
        model=MODEL,
        src=requests,
    )

    job_id = batch_job.name
    print(f"\n  ✓ Batch job submitted!")
    print(f"  Job ID: {job_id}")
    print(f"  State:  {batch_job.state}")

    # Save job ID and metadata
    JOB_FILE.write_text(job_id)
    meta_path = Path("data/gemini_batch_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    print(f"\n  Job ID saved to: {JOB_FILE}")
    print(f"  Metadata saved to: {meta_path}")
    print(f"\n  Run 'python run_gemini_batch.py --poll' to check status.")
    print(f"  Results typically ready within 1–24 hours.")

def poll_and_write():
    if not JOB_FILE.exists():
        print("ERROR: No job ID found. Run without --poll first to submit.")
        sys.exit(1)

    job_id = JOB_FILE.read_text().strip()
    meta_path = Path("data/gemini_batch_metadata.json")
    if not meta_path.exists():
        print("ERROR: Metadata file missing.")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"Checking batch job: {job_id}")
    batch_job = client.batches.get(name=job_id)
    print(f"State: {batch_job.state}")

    state = str(batch_job.state)
    if "SUCCEEDED" not in state and "COMPLETED" not in state and "SUCCESS" not in state:
        print(f"Not ready yet — state: {state}")
        if hasattr(batch_job, 'update_time'):
            print(f"Last updated: {batch_job.update_time}")
        print("Try again later with: python run_gemini_batch.py --poll")
        return

    print("Job complete! Writing results to CSVs...")

    # Load all dataframes
    dfs = {}
    for name in DATASETS:
        path = SAMPLE_DIR / f"sample_{name}.csv"
        if path.exists():
            df = pd.read_csv(path, dtype=str).fillna("")
            if COL not in df.columns:
                df[COL] = ""
            dfs[name] = df

    # Write results
    inlined = batch_job.dest.inlined_responses or []
    written = 0
    for i, inlined_resp in enumerate(inlined):
        if i >= len(metadata):
            break
        meta = metadata[i]
        name = meta["dataset"]
        idx  = meta["index"]
        try:
            raw = inlined_resp.response.candidates[0].content.parts[0].text or ""
        except Exception:
            raw = ""
        pred = normalise(raw.strip())
        dfs[name].at[idx, COL] = pred
        written += 1

    # Save all CSVs
    for name, df in dfs.items():
        path = SAMPLE_DIR / f"sample_{name}.csv"
        df.to_csv(path, index=False)
        done = (df[COL] != "").sum()
        print(f"  {name:<12} {done}/{len(df)} filled")

    print(f"\n  ✓ {written} predictions written — {datetime.now():%Y-%m-%d %H:%M}")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--poll" in sys.argv:
        poll_and_write()
    else:
        submit_batch()
