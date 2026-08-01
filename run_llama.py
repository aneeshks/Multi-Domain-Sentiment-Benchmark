#!/usr/bin/env python3
"""
run_llama.py
══════════════════════════════════════════════════════════════════════
Run Llama 3.1 (via Ollama) over all sampled datasets and write
predictions back as a new column (pred_llama31) in each sample CSV.

Fully resumable — skips rows that already have a prediction.
Runs locally — FREE, no API key needed.

Usage:
    python run_llama.py                        # all 5 datasets
    DATASETS=FPB,SST2 python run_llama.py      # subset
    WORKERS=4 python run_llama.py              # adjust parallelism
    DRYRUN=1 python run_llama.py               # no API calls

Requirements:
    pip install requests python-dotenv pandas
    Ollama running: ollama serve
    Model pulled:   ollama pull llama3.1
"""

import os, re, time, threading, logging, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL       = "llama3.1"
COL         = "pred_llama31"
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")
WORKERS     = int(os.getenv("WORKERS", "4"))
MAX_RETRIES = int(os.getenv("RETRIES", "3"))
DRYRUN      = bool(os.getenv("DRYRUN"))
SAMPLE_DIR  = Path("data/sampled")
DATASETS    = ["FPB", "SST2", "TweetEval", "TFNS", "VADER"]

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

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
_print_lock = threading.Lock()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def normalise(raw: str) -> str:
    for tok in re.findall(r"[a-z]+", raw.lower()):
        for label, words in SYNONYMS.items():
            if tok in words:
                return label
    return "UNPARSED"

def call_llama(sentence: str) -> str:
    """Call Ollama API, return normalised prediction. Retries on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL,
                    "stream": False,
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f'Text: "{sentence}"\nSentiment:'},
                    ],
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()
            return normalise(raw)
        except requests.exceptions.ConnectionError:
            log.error("Cannot connect to Ollama — is it running? (ollama serve)")
            return "ERROR"
        except Exception as e:
            log.warning(f"Error attempt {attempt+1}/{MAX_RETRIES}: {e}")
            time.sleep(2 ** attempt)
    return "ERROR"

# ─── Per-dataset run ──────────────────────────────────────────────────────────
def run_dataset(name: str) -> None:
    path = SAMPLE_DIR / f"sample_{name}.csv"
    if not path.exists():
        log.error(f"{path} not found — run prepare_samples.py first")
        return

    df = pd.read_csv(path, dtype=str).fillna("")

    if COL not in df.columns:
        df[COL] = ""

    pending_idx = df.index[df[COL] == ""].tolist()
    total   = len(df)
    skipped = total - len(pending_idx)

    print(f"\n{'═'*65}")
    print(f"▶  {name}  |  {total} rows  |  done: {skipped}  |  todo: {len(pending_idx)}")
    print(f"{'═'*65}")

    if not pending_idx:
        print(f"   ✓ {name} already complete.")
        return

    if DRYRUN:
        print(f"   DRYRUN — would process {len(pending_idx)} rows")
        return

    counter = {"n": 0, "ok": 0, "bad": 0, "err": 0}
    start   = time.time()
    lock    = threading.Lock()

    def work(idx):
        sentence = df.at[idx, "sentence"]
        pred     = call_llama(sentence)
        with lock:
            df.at[idx, COL] = pred
        return idx, pred, df.at[idx, "human_label"]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(work, i): i for i in pending_idx}
        for fut in as_completed(futures):
            idx, pred, gold = fut.result()
            with _print_lock:
                counter["n"] += 1
                if pred == "ERROR":
                    counter["err"] += 1
                    if counter["err"] == 1:
                        log.error("First ERROR — check Ollama is running (ollama serve)")
                elif pred == gold:
                    counter["ok"] += 1
                else:
                    counter["bad"] += 1

                n = counter["n"]
                if n % 50 == 0 or n == len(pending_idx):
                    elapsed = time.time() - start
                    rate    = n / elapsed if elapsed else 0
                    eta     = (len(pending_idx) - n) / rate if rate else 0
                    acc     = counter["ok"] / max(counter["ok"] + counter["bad"], 1)
                    print(f"   [{n:>5}/{len(pending_idx)}]  acc={acc:.3f}  "
                          f"err={counter['err']}  {rate:.1f} calls/s  ETA {eta/60:.0f}m")

            if counter["n"] % 100 == 0:
                with lock:
                    df.to_csv(path, index=False)

    df.to_csv(path, index=False)
    acc = counter["ok"] / max(counter["ok"] + counter["bad"], 1)
    print(f"\n   ✓ {name} done — acc={acc:.3f}  err={counter['err']}  → {path}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Check Ollama is reachable
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(MODEL in m for m in models):
            log.warning(f"{MODEL} not found in Ollama. Run: ollama pull {MODEL}")
        else:
            log.info(f"Ollama ready — {MODEL} found")
    except Exception:
        log.error(f"Cannot reach Ollama at {OLLAMA_URL} — run: ollama serve")
        sys.exit(1)

    selected = os.getenv("DATASETS")
    names    = [n.strip() for n in selected.split(",")] if selected else DATASETS

    print("═" * 65)
    print(f"Llama Sentiment Run — {MODEL} via Ollama")
    print(f"  datasets : {names}")
    print(f"  workers  : {WORKERS}  |  max_retries: {MAX_RETRIES}"
          f"{'  |  DRYRUN' if DRYRUN else ''}")
    print(f"  ollama   : {OLLAMA_URL}  |  FREE (local)")
    print("═" * 65)

    for name in names:
        run_dataset(name)

    print(f"\n{'═'*65}\nALL DONE — {datetime.now():%Y-%m-%d %H:%M}\n{'═'*65}")

    for name in names:
        path = SAMPLE_DIR / f"sample_{name}.csv"
        if path.exists():
            df = pd.read_csv(path, dtype=str).fillna("")
            if COL in df.columns:
                done = (df[COL] != "").sum()
                dist = df[df[COL] != ""][COL].value_counts().to_dict()
                print(f"  {name:<12} {done}/{len(df)} filled  |  {dist}")

if __name__ == "__main__":
    main()
