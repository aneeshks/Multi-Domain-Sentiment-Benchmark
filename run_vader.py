#!/usr/bin/env python3
"""
run_vader.py
══════════════════════════════════════════════════════════════════════
Run VADER (vaderSentiment 3.3.2) over all sampled datasets and write
predictions back as a new column (pred_vader) in each sample CSV.

VADER uses compound score thresholds:
    compound >= 0.05  → positive
    compound <= -0.05 → negative
    otherwise         → neutral

Fully resumable. Runs locally — FREE, instant, no API.

Usage:
    python run_vader.py
    DATASETS=FPB,SST2 python run_vader.py

Requirements:
    pip install vaderSentiment==3.3.2 pandas
"""

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── Config ───────────────────────────────────────────────────────────────────
COL        = "pred_vader"
SAMPLE_DIR = Path("data/sampled")
DATASETS   = ["FPB", "SST2", "TweetEval", "TFNS", "VADER"]

# Standard VADER thresholds
POS_THRESHOLD =  0.05
NEG_THRESHOLD = -0.05

analyser = SentimentIntensityAnalyzer()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def classify(sentence: str) -> str:
    score = analyser.polarity_scores(sentence)["compound"]
    if score >= POS_THRESHOLD:
        return "positive"
    elif score <= NEG_THRESHOLD:
        return "negative"
    else:
        return "neutral"

# ─── Per-dataset run ──────────────────────────────────────────────────────────
def run_dataset(name: str) -> None:
    path = SAMPLE_DIR / f"sample_{name}.csv"
    if not path.exists():
        print(f"  ERROR: {path} not found")
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

    ok = bad = 0
    for idx in pending_idx:
        pred = classify(df.at[idx, "sentence"])
        df.at[idx, COL] = pred
        gold = df.at[idx, "human_label"]
        if pred == gold:
            ok += 1
        else:
            bad += 1

    df.to_csv(path, index=False)
    acc = ok / max(ok + bad, 1)
    print(f"   ✓ {name} done — acc={acc:.3f}  → {path}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    selected = os.getenv("DATASETS")
    names    = [n.strip() for n in selected.split(",")] if selected else DATASETS

    print("═" * 65)
    print(f"VADER Sentiment Run — vaderSentiment 3.3.2")
    print(f"  datasets  : {names}")
    print(f"  thresholds: pos>={POS_THRESHOLD}  neg<={NEG_THRESHOLD}")
    print(f"  FREE — runs locally, no API calls")
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
