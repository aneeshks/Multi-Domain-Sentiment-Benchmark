#!/usr/bin/env python3
"""
run_openai.py
══════════════════════════════════════════════════════════════════════
Run GPT-4o and GPT-4.1 over all sampled datasets and write predictions
back as new columns (pred_gpt4o, pred_gpt41) in each sample CSV.

Fully resumable — skips rows that already have a prediction.

Usage:
    python run_openai.py                        # both models, all datasets
    MODELS=gpt-4o python run_openai.py          # single model
    DATASETS=FPB,SST2 python run_openai.py      # subset of datasets
    WORKERS=8 python run_openai.py              # more parallelism
    DRYRUN=1 python run_openai.py               # no API calls

Requirements:
    pip install openai python-dotenv pandas
    OPENAI_API_KEY in .env
"""

import os, re, time, threading, logging, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from openai import OpenAI, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
MODELS = {
    "gpt-4o":  "pred_gpt4o",
    "gpt-5.5": "pred_gpt55",
}
WORKERS     = int(os.getenv("WORKERS", "5"))
MAX_RETRIES = int(os.getenv("RETRIES", "5"))
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

# ─── OpenAI client ────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Helpers ──────────────────────────────────────────────────────────────────
def normalise(raw: str) -> str:
    for tok in re.findall(r"[a-z]+", raw.lower()):
        for label, words in SYNONYMS.items():
            if tok in words:
                return label
    return "UNPARSED"

def call_openai(sentence: str, model: str) -> str:
    """Call OpenAI API, return normalised prediction. Retries with backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            # gpt-5.x uses reasoning tokens, needs more headroom
            max_tok = 100 if model.startswith("gpt-5") else 10
            params = {
                "model": model,
                "max_completion_tokens": max_tok,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f'Text: "{sentence}"\nSentiment:'},
                ],
            }
            # gpt-5.5+ does not support temperature parameter
            if not model.startswith("gpt-5"):
                params["temperature"] = 0
            resp = client.chat.completions.create(**params)
            raw = resp.choices[0].message.content.strip()
            return normalise(raw)
        except RateLimitError:
            wait = 2 ** (attempt + 2)
            log.warning(f"Rate limit — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
        except APIStatusError as e:
            log.warning(f"API error {e.status_code} — attempt {attempt+1}/{MAX_RETRIES}")
            time.sleep(2 ** attempt)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            time.sleep(2 ** attempt)
    return "ERROR"

# ─── Per-dataset run ──────────────────────────────────────────────────────────
def run_dataset(name: str, model: str, col: str) -> None:
    path = SAMPLE_DIR / f"sample_{name}.csv"
    if not path.exists():
        log.error(f"{path} not found — run prepare_samples.py first")
        return

    df = pd.read_csv(path, dtype=str).fillna("")

    if col not in df.columns:
        df[col] = ""

    pending_idx = df.index[df[col] == ""].tolist()
    total   = len(df)
    skipped = total - len(pending_idx)

    print(f"\n{'═'*65}")
    print(f"▶  {name} [{model}]  |  {total} rows  |  done: {skipped}  |  todo: {len(pending_idx)}")
    print(f"{'═'*65}")

    if not pending_idx:
        print(f"   ✓ {name} [{model}] already complete.")
        return

    if DRYRUN:
        print(f"   DRYRUN — would call API for {len(pending_idx)} rows")
        return

    counter = {"n": 0, "ok": 0, "bad": 0, "err": 0}
    start   = time.time()
    lock    = threading.Lock()

    def work(idx):
        sentence = df.at[idx, "sentence"]
        pred     = call_openai(sentence, model)
        with lock:
            df.at[idx, col] = pred
        return idx, pred, df.at[idx, "human_label"]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(work, i): i for i in pending_idx}
        for fut in as_completed(futures):
            idx, pred, gold = fut.result()
            with _print_lock:
                counter["n"] += 1
                if pred == "ERROR":
                    counter["err"] += 1
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
    print(f"\n   ✓ {name} [{model}] done — acc={acc:.3f}  err={counter['err']}  → {path}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    selected_datasets = os.getenv("DATASETS")
    names = [n.strip() for n in selected_datasets.split(",")] if selected_datasets else DATASETS

    selected_models = os.getenv("MODELS")
    models = {}
    if selected_models:
        for m in selected_models.split(","):
            m = m.strip()
            if m in MODELS:
                models[m] = MODELS[m]
    else:
        models = MODELS

    print("═" * 65)
    print(f"OpenAI Sentiment Run — {list(models.keys())}")
    print(f"  datasets : {names}")
    print(f"  workers  : {WORKERS}  |  max_retries: {MAX_RETRIES}"
          f"{'  |  DRYRUN' if DRYRUN else ''}")
    print("═" * 65)

    for model, col in models.items():
        for name in names:
            run_dataset(name, model, col)

    print(f"\n{'═'*65}\nALL DONE — {datetime.now():%Y-%m-%d %H:%M}\n{'═'*65}")

    for name in names:
        path = SAMPLE_DIR / f"sample_{name}.csv"
        if path.exists():
            df = pd.read_csv(path, dtype=str).fillna("")
            parts = []
            for model, col in models.items():
                if col in df.columns:
                    done = (df[col] != "").sum()
                    dist = df[df[col] != ""][col].value_counts().to_dict()
                    parts.append(f"{model}: {done}/{len(df)} {dist}")
            if parts:
                print(f"  {name:<12} | {' | '.join(parts)}")

if __name__ == "__main__":
    main()
