#!/usr/bin/env python3
"""
prepare_samples.py
══════════════════════════════════════════════════════════════════════
Generate and save fixed 2K random samples from each dataset to
data/sampled/. Run this ONCE before the main experiment.

Outputs (one CSV per dataset):
    data/sampled/sample_FPB.csv
    data/sampled/sample_SST2.csv
    data/sampled/sample_TweetEval.csv
    data/sampled/sample_TFNS.csv
    data/sampled/sample_VADER.csv

Schema (identical across all files):
    id, dataset, sentence, human_label, human_score, source

Usage:
    python prepare_samples.py
"""

import random
from pathlib import Path
import pandas as pd

SEED       = 42
SAMPLE_CAP = 2000
DATA_DIR   = Path("data")
OUT_DIR    = Path("data/sampled")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def seeded_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Return up to n rows, randomly sampled with a fixed seed, original order preserved."""
    if len(df) <= n:
        return df.reset_index(drop=True)
    idx = sorted(random.Random(seed).sample(range(len(df)), n))
    return df.iloc[idx].reset_index(drop=True)

def assign_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "id", [f"{prefix}_{i:05d}" for i in range(len(df))])
    return df

def save(df: pd.DataFrame, name: str):
    path = OUT_DIR / f"sample_{name}.csv"
    df.to_csv(path, index=False)
    label_dist = df["human_label"].value_counts().to_dict()
    print(f"  ✓ {name:<12} {len(df):>5} rows  →  {path.name}  |  {label_dist}")

# ─── Loaders ──────────────────────────────────────────────────────────────────
def load_fpb() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "D1_FinancialPhraseBank/fpb_experiment.csv")
    df = df.rename(columns={"text": "sentence", "study_label": "human_label"})
    df["human_score"] = ""
    df["source"] = "Financial PhraseBank (all-agree)"
    df["dataset"] = "FPB"
    return df[["dataset", "sentence", "human_label", "human_score", "source"]]

def load_sst2() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "D2_SST2/sst2_experiment.csv")
    df = df.rename(columns={"text": "sentence", "study_label": "human_label"})
    df["human_score"] = ""
    df["source"] = "SST-2 validation split"
    df["dataset"] = "SST2"
    return df[["dataset", "sentence", "human_label", "human_score", "source"]]

def load_tweeteval() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "D3_TweetEval/tweeteval_experiment.csv")
    df = df.rename(columns={"text": "sentence", "study_label": "human_label"})
    df["human_score"] = ""
    df["source"] = "TweetEval sentiment test split"
    df["dataset"] = "TweetEval"
    return df[["dataset", "sentence", "human_label", "human_score", "source"]]

def load_tfns() -> pd.DataFrame:
    train = pd.read_csv(DATA_DIR / "D4_TFNS/tfns_train.csv")
    val   = pd.read_csv(DATA_DIR / "D4_TFNS/tfns_validation.csv")
    df = pd.concat([train, val], ignore_index=True)
    df = df.rename(columns={"text": "sentence", "study_label": "human_label"})
    df["human_score"] = ""
    df["source"] = "TFNS train+validation"
    df["dataset"] = "TFNS"
    return df[["dataset", "sentence", "human_label", "human_score", "source"]]

def load_vader() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "D5_VADER/vader_experiment.csv")
    df = df.rename(columns={"text": "sentence", "study_label": "human_label", "score": "human_score"})
    df["source"] = "VADER validation (" + df["source"] + ")"
    df["dataset"] = "VADER"
    return df[["dataset", "sentence", "human_label", "human_score", "source"]]

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating samples (seed={SEED}, cap={SAMPLE_CAP} per dataset)\n")

    datasets = {
        "FPB":       (load_fpb,       "FPB"),
        "SST2":      (load_sst2,      "SST2"),
        "TweetEval": (load_tweeteval, "TE"),
        "TFNS":      (load_tfns,      "TFNS"),
        "VADER":     (load_vader,     "VADER"),
    }

    total = 0
    for name, (loader, prefix) in datasets.items():
        df = loader()
        df = seeded_sample(df, SAMPLE_CAP, SEED)
        df = assign_ids(df, prefix)
        save(df, name)
        total += len(df)

    print(f"\nTotal sentences across all datasets: {total:,}")
    print(f"Total API calls needed (6 models):   {total * 6:,}")

if __name__ == "__main__":
    main()
