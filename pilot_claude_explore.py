#!/usr/bin/env python3
"""
pilot_claude_explore.py
────────────────────────────────────────────────────────────────────────────────
Runs all 8 Claude models on SAMPLE_SIZE sentences per dataset.
FOR PERSONAL ANALYSIS ONLY — not for the paper.

Usage:
    python pilot_claude_explore.py          # pilot: 10 sentences per dataset
    FULL=1 python pilot_claude_explore.py   # full run: all sentences

Outputs:
    predictions_claude_explore.csv          # all predictions
    run_log.txt                             # timestamped log

Requirements:
    pip install anthropic datasets vaderSentiment python-dotenv pandas --break-system-packages
"""

import os, sys, time, re, logging
from datetime import datetime
from pathlib import Path

import subprocess
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
SAMPLE_SIZE   = None if os.getenv("FULL") else 10   # 10 for pilot, None = full
DELAY         = 1.0          # seconds between API calls (be safe with rate limits)
MAX_RETRIES   = 5            # retries on rate-limit / timeout errors
OUTPUT_FILE   = "predictions_claude_explore.csv"
LOG_FILE      = "run_log.txt"

CLAUDE_MODELS = [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
]

SYSTEM_PROMPT = """You are a sentiment classifier. Classify the sentiment of the following text using these definitions:
  positive — the text expresses optimism, approval, or good news
  negative — the text expresses pessimism, criticism, or bad news
  neutral  — the text is factual, balanced, or does not lean either way
Reply with exactly one word: positive, negative, or neutral."""

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─── Output normalisation ─────────────────────────────────────────────────────
SYNONYMS = {
    "positive": ["positive", "bullish", "optimistic", "favorable", "good", "pos"],
    "negative": ["negative", "bearish", "pessimistic", "unfavorable", "bad", "neg"],
    "neutral":  ["neutral", "factual", "balanced", "mixed", "neu"],
}

def normalise(raw: str) -> str:
    """Convert raw LLM output to positive / negative / neutral."""
    text = raw.strip().lower()
    for label, synonyms in SYNONYMS.items():
        for s in synonyms:
            if s in text:
                return label
    log.warning(f"Could not normalise output: '{raw}' — defaulting to neutral")
    return "neutral"

# ─── Claude CLI call ──────────────────────────────────────────────────────────
def call_claude(model: str, sentence: str) -> str:
    """Call Claude via CLI (uses Claude Max subscription — no API credits needed)."""
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f'Text: "{sentence}"\n'
        f"Sentiment:"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--model", model],
                capture_output=True,
                text=True,
                timeout=30,
            )
            raw = result.stdout.strip()
            if raw:
                return normalise(raw)
            if result.stderr:
                log.warning(f"stderr: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            log.warning(f"Timeout (attempt {attempt}/{MAX_RETRIES})")
        except Exception as e:
            log.error(f"Unexpected error: {e}")
        time.sleep(2 ** attempt)
    return "ERROR"

# ─── Dataset loading ──────────────────────────────────────────────────────────
def load_datasets(sample_size):
    """Load all 5 datasets. Returns list of dicts with id, dataset, sentence, human_label."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Run: pip install datasets --break-system-packages")
        sys.exit(1)

    rows = []

    # D1 — Financial PhraseBank (all-agree config)
    log.info("Loading D1 — Financial PhraseBank...")
    try:
        ds = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree",
                          trust_remote_code=True, split="train")
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        data = [(r["sentence"], label_map[r["label"]]) for r in ds]
        if sample_size:
            data = data[:sample_size]
        for i, (sent, label) in enumerate(data):
            rows.append({"id": f"D1_{i:05d}", "dataset": "FPB", "sentence": sent, "human_label": label})
        log.info(f"  D1 loaded: {len(data)} sentences")
    except Exception as e:
        log.error(f"  D1 failed: {e}")

    # D2 — SST-2 (validation split)
    log.info("Loading D2 — SST-2...")
    try:
        ds = load_dataset("stanfordnlp/sst2", split="validation")
        label_map = {0: "negative", 1: "positive"}
        data = [(r["sentence"], label_map[r["label"]]) for r in ds]
        if sample_size:
            data = data[:sample_size]
        for i, (sent, label) in enumerate(data):
            rows.append({"id": f"D2_{i:05d}", "dataset": "SST2", "sentence": sent, "human_label": label})
        log.info(f"  D2 loaded: {len(data)} sentences")
    except Exception as e:
        log.error(f"  D2 failed: {e}")

    # D3 — TweetEval sentiment (test split)
    log.info("Loading D3 — TweetEval...")
    try:
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        data = [(r["text"], label_map[r["label"]]) for r in ds]
        if sample_size:
            data = data[:sample_size]
        for i, (sent, label) in enumerate(data):
            rows.append({"id": f"D3_{i:05d}", "dataset": "TweetEval", "sentence": sent, "human_label": label})
        log.info(f"  D3 loaded: {len(data)} sentences")
    except Exception as e:
        log.error(f"  D3 failed: {e}")

    # D4 — Twitter Financial News Sentiment
    log.info("Loading D4 — TFNS...")
    try:
        ds_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        ds_val   = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
        # Labels: 0=Bearish(negative), 1=Bullish(positive), 2=Neutral
        label_map = {0: "negative", 1: "positive", 2: "neutral"}
        data = [(r["text"], label_map[r["label"]]) for r in list(ds_train) + list(ds_val)]
        if sample_size:
            data = data[:sample_size]
        for i, (sent, label) in enumerate(data):
            rows.append({"id": f"D4_{i:05d}", "dataset": "TFNS", "sentence": sent, "human_label": label})
        log.info(f"  D4 loaded: {len(data)} sentences")
    except Exception as e:
        log.error(f"  D4 failed: {e}")

    # D5 — VADER validation set (download from GitHub)
    log.info("Loading D5 — VADER validation set...")
    try:
        import urllib.request
        vader_url = "https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/additional_resources/vader_icwsm2014_data.txt"
        vader_path = Path("vader_icwsm2014_data.txt")
        if not vader_path.exists():
            log.info("  Downloading VADER validation data...")
            urllib.request.urlretrieve(vader_url, vader_path)
        data = []
        with open(vader_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        score = float(parts[0])
                        text  = parts[-1]
                        if score >= 0.05:
                            label = "positive"
                        elif score <= -0.05:
                            label = "negative"
                        else:
                            label = "neutral"
                        data.append((text, label))
                    except ValueError:
                        continue
        if sample_size:
            data = data[:sample_size]
        for i, (sent, label) in enumerate(data):
            rows.append({"id": f"D5_{i:05d}", "dataset": "VADER", "sentence": sent, "human_label": label})
        log.info(f"  D5 loaded: {len(data)} sentences")
    except Exception as e:
        log.error(f"  D5 failed: {e}")

    log.info(f"Total sentences loaded: {len(rows)}")
    return rows

# ─── Checkpoint helpers ───────────────────────────────────────────────────────
def load_checkpoint(output_file: str) -> pd.DataFrame:
    """Load existing results so we can skip already-done calls."""
    if Path(output_file).exists():
        df = pd.read_csv(output_file)
        log.info(f"Checkpoint found: {len(df)} rows already saved")
        return df
    return pd.DataFrame()

def already_done(checkpoint_df: pd.DataFrame, sentence_id: str, model: str) -> bool:
    if checkpoint_df.empty:
        return False
    # A row only counts as done if it has a real (non-ERROR) prediction,
    # so failed calls get retried on resume instead of being skipped forever.
    match = ((checkpoint_df["id"] == sentence_id) &
             (checkpoint_df["model"] == model) &
             (checkpoint_df["prediction"] != "ERROR"))
    return match.any()

def save_row(output_file: str, row: dict):
    """Append one row to the CSV (creates file with header if needed)."""
    df = pd.DataFrame([row])
    write_header = not Path(output_file).exists()
    df.to_csv(output_file, mode="a", header=write_header, index=False)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info(f"Pilot Claude Explore — started {datetime.now()}")
    log.info(f"Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'FULL RUN'}")
    log.info(f"Models: {len(CLAUDE_MODELS)}")
    log.info("=" * 60)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not found in .env — aborting")
        sys.exit(1)

    # Load datasets
    sentences = load_datasets(SAMPLE_SIZE)
    if not sentences:
        log.error("No sentences loaded — check dataset connections")
        sys.exit(1)

    # Load checkpoint
    checkpoint = load_checkpoint(OUTPUT_FILE)

    # Count work to do
    total_calls = len(sentences) * len(CLAUDE_MODELS)
    done_calls  = len(checkpoint) if not checkpoint.empty else 0
    log.info(f"Total API calls needed: {total_calls} | Already done: {done_calls}")

    # Run
    total_to_do = sum(
        1 for s in sentences for m in CLAUDE_MODELS
        if not already_done(checkpoint, s["id"], m)
    )
    call_count = 0

    for sent_dict in sentences:
        print(f"\n{'─'*60}")
        print(f"📄 Sentence [{sent_dict['id']}] — {sent_dict['dataset']}")
        print(f"   Text   : {sent_dict['sentence'][:80]}{'...' if len(sent_dict['sentence'])>80 else ''}")
        print(f"   Human  : {sent_dict['human_label'].upper()}")
        print(f"{'─'*60}")

        for model in CLAUDE_MODELS:
            if already_done(checkpoint, sent_dict["id"], model):
                print(f"   ✓ {model:<40} [already done — skipped]")
                continue

            call_count += 1
            short_model = model.replace("claude-", "")
            print(f"   ⏳ [{call_count}/{total_to_do}] {short_model:<38} ", end="", flush=True)

            prediction = call_claude(model, sent_dict["sentence"])

            emoji = "✅" if prediction == sent_dict["human_label"] else ("❌" if prediction != "ERROR" else "💥")
            print(f"→ {prediction.upper():<10} {emoji}")

            row = {
                "id":           sent_dict["id"],
                "dataset":      sent_dict["dataset"],
                "sentence":     sent_dict["sentence"],
                "human_label":  sent_dict["human_label"],
                "model":        model,
                "prediction":   prediction,
                "timestamp":    datetime.now().isoformat(),
            }
            save_row(OUTPUT_FILE, row)
            time.sleep(DELAY)

    log.info("=" * 60)
    log.info(f"Done! Results saved to {OUTPUT_FILE}")
    log.info(f"Total new API calls made: {call_count}")
    log.info("=" * 60)

    # Quick summary
    df = pd.read_csv(OUTPUT_FILE)
    print("\n--- Quick Summary ---")
    print(f"Rows in output: {len(df)}")
    print(f"\nPrediction distribution:\n{df['prediction'].value_counts()}")
    print(f"\nRows per model:\n{df['model'].value_counts()}")

if __name__ == "__main__":
    main()
